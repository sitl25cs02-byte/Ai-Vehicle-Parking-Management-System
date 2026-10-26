"""
Vehicle Parking Management System — v2 "Control Center"
=========================================================
A full-stack Flask application featuring:
  - User & Admin authentication (session based)
  - Parking slot management (Admin)
  - AI-based number-plate detection on booking (OpenCV + Tesseract OCR)
  - Smart Slot Allocation (rule-based load-balancing algorithm)
  - Duplicate / fraud detection (same vehicle, two active bookings)
  - Overstay detection & alerts
  - Advance reservations + check-in / cancel flow
  - QR digital parking pass + camera-based scan/verify (client-side)
  - Analytics dashboard (revenue, occupancy, vehicle split, peak hours)
  - Vehicle history search
  - Automatic billing on checkout

Run with:  python app.py
Default admin login -> username: admin | password: admin123
"""
import os
import math
import uuid
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import db
import services
from detection.plate_detector import detect_plate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers & decorators
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrapper


def calculate_amount(entry_time: datetime, exit_time: datetime, rate_per_hour: float) -> float:
    duration_seconds = (exit_time - entry_time).total_seconds()
    hours = max(1, math.ceil(duration_seconds / 3600))
    return round(hours * rate_per_hour, 2)


@app.context_processor
def inject_globals():
    return {"now": datetime.now()}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()
        phone = request.form.get("phone", "").strip()

        if not username or not password or not full_name:
            flash("Please fill all required fields.", "danger")
            return redirect(url_for("register"))

        conn = db.get_connection()
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            conn.close()
            flash("Username already taken. Please choose another.", "danger")
            return redirect(url_for("register"))

        conn.execute(
            "INSERT INTO users (username, password_hash, full_name, phone, role, created_at) "
            "VALUES (?, ?, ?, ?, 'user', ?)",
            (username, generate_password_hash(password), full_name, phone,
             datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        conn = db.get_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Dashboard (role-aware)
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    conn = db.get_connection()
    if session["role"] == "admin":
        total_slots = conn.execute("SELECT COUNT(*) c FROM slots").fetchone()["c"]
        occupied = conn.execute("SELECT COUNT(*) c FROM slots WHERE status='occupied'").fetchone()["c"]
        reserved = conn.execute("SELECT COUNT(*) c FROM slots WHERE status='reserved'").fetchone()["c"]
        available = total_slots - occupied - reserved
        active_bookings = conn.execute(
            "SELECT COUNT(*) c FROM bookings WHERE status='active'"
        ).fetchone()["c"]
        revenue_row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) r FROM bookings WHERE status='completed'"
        ).fetchone()
        total_revenue = revenue_row["r"]
        overstay_list = services.get_overstay_bookings(conn)
        recent_bookings = conn.execute(
            """SELECT b.*, s.slot_number, u.full_name FROM bookings b
               JOIN slots s ON b.slot_id = s.id
               JOIN users u ON b.user_id = u.id
               ORDER BY b.id DESC LIMIT 5"""
        ).fetchall()
        conn.close()
        return render_template(
            "dashboard_admin.html",
            total_slots=total_slots, occupied=occupied, available=available,
            reserved=reserved, active_bookings=active_bookings, total_revenue=total_revenue,
            recent_bookings=recent_bookings, overstay_count=len(overstay_list),
            overstay_list=overstay_list[:5],
        )
    else:
        my_active = conn.execute(
            """SELECT b.*, s.slot_number, s.rate_per_hour FROM bookings b
               JOIN slots s ON b.slot_id = s.id
               WHERE b.user_id = ? AND b.status = 'active'""",
            (session["user_id"],),
        ).fetchall()
        my_reserved = conn.execute(
            """SELECT b.*, s.slot_number, s.rate_per_hour FROM bookings b
               JOIN slots s ON b.slot_id = s.id
               WHERE b.user_id = ? AND b.status = 'reserved'""",
            (session["user_id"],),
        ).fetchall()
        available_slots = conn.execute(
            "SELECT COUNT(*) c FROM slots WHERE status='available'"
        ).fetchone()["c"]
        total_spent_row = conn.execute(
            "SELECT COALESCE(SUM(amount),0) t FROM bookings WHERE user_id=? AND status='completed'",
            (session["user_id"],),
        ).fetchone()
        conn.close()
        my_active = [dict(b, elapsed=services.elapsed_hours(b["entry_time"])) for b in my_active]
        return render_template(
            "dashboard_user.html",
            my_active=my_active, my_reserved=my_reserved, available_slots=available_slots,
            total_spent=total_spent_row["t"],
        )


# ---------------------------------------------------------------------------
# Admin: Slot management
# ---------------------------------------------------------------------------
@app.route("/admin/slots")
@login_required
@admin_required
def admin_slots():
    conn = db.get_connection()
    slots = conn.execute("SELECT * FROM slots ORDER BY floor, slot_number").fetchall()
    conn.close()
    return render_template("admin_slots.html", slots=slots)


@app.route("/admin/slots/add", methods=["POST"])
@login_required
@admin_required
def admin_add_slot():
    slot_number = request.form.get("slot_number", "").strip().upper()
    floor = request.form.get("floor", "").strip()
    vehicle_type = request.form.get("vehicle_type")
    rate = request.form.get("rate_per_hour")

    if not slot_number or not floor or not vehicle_type or not rate:
        flash("All fields are required to add a slot.", "danger")
        return redirect(url_for("admin_slots"))

    conn = db.get_connection()
    existing = conn.execute("SELECT id FROM slots WHERE slot_number = ?", (slot_number,)).fetchone()
    if existing:
        conn.close()
        flash(f"Slot {slot_number} already exists.", "danger")
        return redirect(url_for("admin_slots"))

    conn.execute(
        "INSERT INTO slots (slot_number, floor, vehicle_type, rate_per_hour, status) "
        "VALUES (?, ?, ?, ?, 'available')",
        (slot_number, floor, vehicle_type, float(rate)),
    )
    conn.commit()
    conn.close()
    flash(f"Slot {slot_number} added successfully.", "success")
    return redirect(url_for("admin_slots"))


@app.route("/admin/slots/<int:slot_id>/delete", methods=["POST"])
@login_required
@admin_required
def admin_delete_slot(slot_id):
    conn = db.get_connection()
    slot = conn.execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()
    if not slot:
        conn.close()
        flash("Slot not found.", "danger")
        return redirect(url_for("admin_slots"))
    if slot["status"] != "available":
        conn.close()
        flash("Cannot delete a slot that is occupied or reserved.", "danger")
        return redirect(url_for("admin_slots"))
    conn.execute("DELETE FROM slots WHERE id=?", (slot_id,))
    conn.commit()
    conn.close()
    flash("Slot deleted.", "success")
    return redirect(url_for("admin_slots"))


@app.route("/admin/bookings")
@login_required
@admin_required
def admin_bookings():
    conn = db.get_connection()
    bookings = conn.execute(
        """SELECT b.*, s.slot_number, s.vehicle_type, u.full_name, u.username
           FROM bookings b
           JOIN slots s ON b.slot_id = s.id
           JOIN users u ON b.user_id = u.id
           ORDER BY b.id DESC"""
    ).fetchall()
    total_revenue = conn.execute(
        "SELECT COALESCE(SUM(amount),0) r FROM bookings WHERE status='completed'"
    ).fetchone()["r"]
    conn.close()
    bookings = [dict(b, overstay=services.is_overstay(b["entry_time"]) if b["status"] == "active" else False)
                for b in bookings]
    return render_template("admin_bookings.html", bookings=bookings, total_revenue=total_revenue)


@app.route("/admin/analytics")
@login_required
@admin_required
def admin_analytics():
    conn = db.get_connection()
    revenue_labels, revenue_values = services.revenue_last_n_days(conn, 14)
    floor_labels, floor_available, floor_occupied, floor_reserved = services.occupancy_by_floor(conn)
    vtype_labels, vtype_counts, vtype_revenue = services.vehicle_type_split(conn)
    hour_labels, hour_counts = services.peak_hour_histogram(conn)
    conn.close()
    return render_template(
        "admin_analytics.html",
        revenue_labels=revenue_labels, revenue_values=revenue_values,
        floor_labels=floor_labels, floor_available=floor_available,
        floor_occupied=floor_occupied, floor_reserved=floor_reserved,
        vtype_labels=vtype_labels, vtype_counts=vtype_counts, vtype_revenue=vtype_revenue,
        hour_labels=hour_labels, hour_counts=hour_counts,
    )


@app.route("/admin/search")
@login_required
@admin_required
def admin_search():
    query = request.args.get("q", "").strip().upper()
    results = []
    if query:
        conn = db.get_connection()
        results = conn.execute(
            """SELECT b.*, s.slot_number, s.floor, u.full_name, u.username
               FROM bookings b
               JOIN slots s ON b.slot_id = s.id
               JOIN users u ON b.user_id = u.id
               WHERE b.vehicle_number LIKE ?
               ORDER BY b.id DESC""",
            (f"%{query}%",),
        ).fetchall()
        conn.close()
    return render_template("admin_search.html", query=query, results=results)


@app.route("/admin/scan")
@login_required
@admin_required
def admin_scan():
    return render_template("admin_scan.html")


@app.route("/admin/api/booking/<token>")
@login_required
@admin_required
def api_booking_by_token(token):
    conn = db.get_connection()
    booking = conn.execute(
        """SELECT b.*, s.slot_number, s.floor, s.rate_per_hour, u.full_name, u.username
           FROM bookings b
           JOIN slots s ON b.slot_id = s.id
           JOIN users u ON b.user_id = u.id
           WHERE b.qr_token = ?""",
        (token,),
    ).fetchone()
    conn.close()
    if not booking:
        return jsonify({"found": False, "message": "No booking matches this QR code."}), 404
    data = dict(booking)
    data["found"] = True
    data["overstay"] = services.is_overstay(booking["entry_time"]) if booking["status"] == "active" else False
    return jsonify(data)


# ---------------------------------------------------------------------------
# User: Browse & book slots
# ---------------------------------------------------------------------------
@app.route("/slots")
@login_required
def view_slots():
    conn = db.get_connection()
    slots = conn.execute("SELECT * FROM slots ORDER BY floor, slot_number").fetchall()
    conn.close()
    floors = {}
    for s in slots:
        floors.setdefault(s["floor"], []).append(s)
    return render_template("slots.html", floors=floors)


@app.route("/auto_book", methods=["POST"])
@login_required
def auto_book():
    vehicle_type = request.form.get("vehicle_type", "car")
    conn = db.get_connection()
    best = services.find_best_slot(conn, vehicle_type)
    conn.close()
    if not best:
        flash(f"No {vehicle_type} slots available right now for Smart Allocation.", "danger")
        return redirect(url_for("view_slots"))
    return redirect(url_for("book_slot", slot_id=best["id"], smart=1))


@app.route("/book/<int:slot_id>", methods=["GET", "POST"])
@login_required
def book_slot(slot_id):
    conn = db.get_connection()
    slot = conn.execute("SELECT * FROM slots WHERE id=?", (slot_id,)).fetchone()

    if not slot:
        conn.close()
        flash("Slot not found.", "danger")
        return redirect(url_for("view_slots"))

    if slot["status"] != "available":
        conn.close()
        flash("This slot is no longer available.", "danger")
        return redirect(url_for("view_slots"))

    if request.method == "POST":
        vehicle_number = request.form.get("vehicle_number", "").strip().upper()
        driver_name = request.form.get("driver_name", "").strip()
        plate_image = request.form.get("plate_image_path", "")
        mode = request.form.get("mode", "now")  # 'now' or 'reserve'
        scheduled_time = request.form.get("scheduled_time", "")

        if not vehicle_number:
            conn.close()
            flash("Vehicle number is required.", "danger")
            return redirect(url_for("book_slot", slot_id=slot_id))

        conflict = services.find_conflicting_booking(conn, vehicle_number)
        if conflict:
            conn.close()
            flash(
                f"⚠️ Fraud check failed: vehicle {vehicle_number} already has an "
                f"{conflict['status']} booking on slot {conflict['slot_number']}.",
                "danger",
            )
            return redirect(url_for("book_slot", slot_id=slot_id))

        qr_token = uuid.uuid4().hex
        created_at = datetime.now().isoformat(timespec="seconds")

        if mode == "reserve":
            if not scheduled_time:
                conn.close()
                flash("Please choose a date & time for your reservation.", "danger")
                return redirect(url_for("book_slot", slot_id=slot_id))
            conn.execute(
                """INSERT INTO bookings
                   (user_id, slot_id, vehicle_number, driver_name, booking_type,
                    scheduled_time, entry_time, status, plate_image, qr_token, created_at)
                   VALUES (?, ?, ?, ?, 'reserved', ?, NULL, 'reserved', ?, ?, ?)""",
                (session["user_id"], slot_id, vehicle_number, driver_name,
                 scheduled_time, plate_image, qr_token, created_at),
            )
            conn.execute("UPDATE slots SET status='reserved' WHERE id=?", (slot_id,))
            conn.commit()
            conn.close()
            flash(f"Slot {slot['slot_number']} reserved for {scheduled_time.replace('T', ' ')}. "
                  f"Check in anytime from 'My Bookings' when you arrive.", "success")
        else:
            now = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """INSERT INTO bookings
                   (user_id, slot_id, vehicle_number, driver_name, booking_type,
                    entry_time, status, plate_image, qr_token, created_at)
                   VALUES (?, ?, ?, ?, 'immediate', ?, 'active', ?, ?, ?)""",
                (session["user_id"], slot_id, vehicle_number, driver_name,
                 now, plate_image, qr_token, created_at),
            )
            conn.execute("UPDATE slots SET status='occupied' WHERE id=?", (slot_id,))
            conn.commit()
            conn.close()
            flash(f"Slot {slot['slot_number']} booked successfully! Entry time recorded.", "success")

        return redirect(url_for("my_bookings"))

    conn.close()
    smart = request.args.get("smart") == "1"
    return render_template("book_slot.html", slot=slot, smart=smart)


@app.route("/detect_plate", methods=["POST"])
@login_required
def detect_plate_endpoint():
    """AJAX endpoint: accepts an uploaded vehicle image, runs the OpenCV +
    Tesseract detection pipeline, and returns the detected plate text."""
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Please upload a valid PNG/JPG image."}), 400

    filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    result = detect_plate(save_path)
    result["image_path"] = f"uploads/{filename}"
    if result.get("cropped_path"):
        result["cropped_url"] = f"uploads/{os.path.basename(result['cropped_path'])}"
    return jsonify(result)


@app.route("/my_bookings")
@login_required
def my_bookings():
    conn = db.get_connection()
    bookings = conn.execute(
        """SELECT b.*, s.slot_number, s.rate_per_hour, s.floor FROM bookings b
           JOIN slots s ON b.slot_id = s.id
           WHERE b.user_id = ?
           ORDER BY b.id DESC""",
        (session["user_id"],),
    ).fetchall()
    conn.close()
    bookings = [dict(b, overstay=services.is_overstay(b["entry_time"]) if b["status"] == "active" else False)
                for b in bookings]
    return render_template("my_bookings.html", bookings=bookings)


@app.route("/checkin/<int:booking_id>", methods=["POST"])
@login_required
def checkin(booking_id):
    conn = db.get_connection()
    booking = conn.execute("SELECT * FROM bookings WHERE id=?", (booking_id,)).fetchone()

    if not booking:
        conn.close()
        flash("Booking not found.", "danger")
        return redirect(url_for("my_bookings"))
    if booking["user_id"] != session["user_id"] and session["role"] != "admin":
        conn.close()
        flash("You are not authorized to check in this booking.", "danger")
        return redirect(url_for("my_bookings"))
    if booking["status"] != "reserved":
        conn.close()
        flash("Only reserved bookings can be checked in.", "warning")
        return redirect(url_for("my_bookings"))

    now = datetime.now().isoformat(timespec="seconds")
    conn.execute("UPDATE bookings SET entry_time=?, status='active' WHERE id=?", (now, booking_id))
    conn.execute("UPDATE slots SET status='occupied' WHERE id=?", (booking["slot_id"],))
    conn.commit()
    conn.close()
    flash("Checked in successfully. Your parking timer has started.", "success")
    return redirect(url_for("my_bookings"))


@app.route("/cancel/<int:booking_id>", methods=["POST"])
@login_required
def cancel_booking(booking_id):
    conn = db.get_connection()
    booking = conn.execute("SELECT * FROM bookings WHERE id=?", (booking_id,)).fetchone()

    if not booking:
        conn.close()
        flash("Booking not found.", "danger")
        return redirect(url_for("my_bookings"))
    if booking["user_id"] != session["user_id"] and session["role"] != "admin":
        conn.close()
        flash("You are not authorized to cancel this booking.", "danger")
        return redirect(url_for("my_bookings"))
    if booking["status"] != "reserved":
        conn.close()
        flash("Only reserved (not yet checked-in) bookings can be cancelled.", "warning")
        return redirect(url_for("my_bookings"))

    conn.execute("UPDATE bookings SET status='cancelled' WHERE id=?", (booking_id,))
    conn.execute("UPDATE slots SET status='available' WHERE id=?", (booking["slot_id"],))
    conn.commit()
    conn.close()
    flash("Reservation cancelled and slot released.", "success")
    return redirect(url_for("my_bookings"))


@app.route("/checkout/<int:booking_id>", methods=["POST"])
@login_required
def checkout(booking_id):
    conn = db.get_connection()
    booking = conn.execute(
        """SELECT b.*, s.rate_per_hour, s.id as slot_pk FROM bookings b
           JOIN slots s ON b.slot_id = s.id
           WHERE b.id = ?""",
        (booking_id,),
    ).fetchone()

    if not booking:
        conn.close()
        flash("Booking not found.", "danger")
        return redirect(url_for("my_bookings"))

    if booking["user_id"] != session["user_id"] and session["role"] != "admin":
        conn.close()
        flash("You are not authorized to checkout this booking.", "danger")
        return redirect(url_for("my_bookings"))

    if booking["status"] != "active":
        conn.close()
        flash("Only active (checked-in) bookings can be checked out.", "warning")
        return redirect(url_for("my_bookings"))

    entry_time = datetime.fromisoformat(booking["entry_time"])
    exit_time = datetime.now()
    amount = calculate_amount(entry_time, exit_time, booking["rate_per_hour"])

    conn.execute(
        "UPDATE bookings SET exit_time=?, status='completed', amount=? WHERE id=?",
        (exit_time.isoformat(timespec="seconds"), amount, booking_id),
    )
    conn.execute("UPDATE slots SET status='available' WHERE id=?", (booking["slot_id"],))
    conn.commit()
    conn.close()

    flash(f"Checked out successfully. Total bill: ₹{amount:.2f}", "success")
    return redirect(url_for("bill_receipt", booking_id=booking_id))


@app.route("/bill/<int:booking_id>")
@login_required
def bill_receipt(booking_id):
    conn = db.get_connection()
    booking = conn.execute(
        """SELECT b.*, s.slot_number, s.floor, s.rate_per_hour, u.full_name, u.username
           FROM bookings b
           JOIN slots s ON b.slot_id = s.id
           JOIN users u ON b.user_id = u.id
           WHERE b.id = ?""",
        (booking_id,),
    ).fetchone()
    conn.close()

    if not booking:
        flash("Bill not found.", "danger")
        return redirect(url_for("my_bookings"))

    if booking["user_id"] != session["user_id"] and session["role"] != "admin":
        flash("Not authorized.", "danger")
        return redirect(url_for("my_bookings"))

    return render_template("bill_receipt.html", booking=booking)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
