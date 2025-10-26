import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client

# ML Libraries for real-time video stream processing
import cv2
import face_recognition

# --- Configuration and Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app) # Allow cross-origin requests, useful for development

# Load Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ENCODINGS_FILE = os.getenv("ENCODINGS_FILE")

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global variables for ML Model
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_IDS = [] # Format: 'RollNo_Name'

# Global variables for real-time attendance tracking
CURRENT_SESSION_ATTENDANCE = {} # Tracks detected students for the current session: {RollNo: Name}

# Load the face recognition model
def load_model():
    """Loads the pre-trained face encodings from the pickle file."""
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_IDS
    if not os.path.exists(ENCODINGS_FILE):
        print(f"CRITICAL: Model file '{ENCODINGS_FILE}' not found. Run train_model.py first!")
        return False
    
    with open(ENCODINGS_FILE, 'rb') as f:
        data = pickle.load(f)
        KNOWN_FACE_ENCODINGS = data["encodings"]
        KNOWN_FACE_IDS = data["names"]
    print(f"ML Model loaded successfully. {len(KNOWN_FACE_IDS)} students registered.")
    return True

# --- Face Recognition Core Logic (Generator Function for Video Stream) ---

def generate_frames():
    """Captures video, detects faces, and generates the video frame stream."""
    camera = cv2.VideoCapture(0) # 0 means default webcam
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    process_this_frame = True

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Optimize processing: only process every other frame
        if process_this_frame:
            # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            current_session_detected = []
            
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Check if the face matches any known face
                matches = face_recognition.compare_faces(KNOWN_FACE_ENCODINGS, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    # Extract the RollNo and Name
                    full_id = KNOWN_FACE_IDS[best_match_index]
                    roll_no, student_name = full_id.split('_', 1)
                    name = student_name.replace(' ', '_')
                    
                    # Mark the student as detected for this session
                    if roll_no not in CURRENT_SESSION_ATTENDANCE:
                        CURRENT_SESSION_ATTENDANCE[roll_no] = student_name.replace('_', ' ')
                        print(f"Attendance marked for: {student_name} (Roll {roll_no})")
                
                # Draw box and label on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        process_this_frame = not process_this_frame

        # Encode the frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in response stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# --- API Endpoints ---

@app.route('/')
def index():
    """Renders the main page (Dashboard/Home)."""
    return render_template('index.html')

@app.route('/student_search')
def student_search():
    """Renders the student search and dashboard page."""
    return render_template('student_search.html')

@app.route('/live_attendance')
def live_attendance():
    """Renders the live camera feed page."""
    return render_template('live_attendance.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint for the streaming video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/finalize_attendance', methods=['POST'])
def finalize_attendance():
    """
    Saves detected students as 'Present' in Supabase.
    Students not detected will be marked 'Absent' after cross-referencing the roster.
    Automatically detects the lecture number for today.
    """
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    # 1. Determine the next lecture number for today
    try:
        # Query to find the highest lecture number for today
        existing_lectures = supabase.table('attendance').select('lecture_number').eq('date', current_date).execute()
        
        if existing_lectures.data:
            # Get the maximum lecture number
            max_lecture = max([record['lecture_number'] for record in existing_lectures.data])
            lecture_number = max_lecture + 1
        else:
            # First lecture of the day
            lecture_number = 1
            
        print(f"Creating attendance for Lecture {lecture_number} on {current_date}")
        
    except Exception as e:
        print(f"Error determining lecture number: {e}")
        return jsonify({"error": f"Could not determine lecture number: {str(e)}"}), 500

    # 2. Fetch the full student roster from Supabase
    try:
        response = supabase.table('students').select('*').execute()
        full_roster = response.data
        
        if not full_roster:
            return jsonify({"error": "No students found in the database. Please add students first."}), 400
            
    except Exception as e:
        print(f"Error fetching roster: {e}")
        return jsonify({"error": f"Could not fetch student roster from Supabase: {str(e)}"}), 500

    attendance_data_to_insert = []
    
    # 3. Iterate through the full roster and determine status
    for student in full_roster:
        roll_no = str(student['roll_no'])
        name = student['name']
        
        status = "Absent"
        if roll_no in CURRENT_SESSION_ATTENDANCE:
            status = "Present"
            
        attendance_data_to_insert.append({
            'roll_no': int(roll_no),
            'date': current_date,
            'time': current_time,
            'lecture_number': lecture_number,
            'status': status
        })

    # 4. Insert all records into the attendance table
    try:
        response = supabase.table('attendance').insert(attendance_data_to_insert).execute()
        # Reset the session attendance tracker
        CURRENT_SESSION_ATTENDANCE.clear()
        
        return jsonify({
            "message": f"Attendance for Lecture {lecture_number} finalized and saved successfully.",
            "lecture_number": lecture_number,
            "date": current_date,
            "present_count": len([d for d in attendance_data_to_insert if d['status'] == 'Present']),
            "absent_count": len([d for d in attendance_data_to_insert if d['status'] == 'Absent']),
        }), 200

    except Exception as e:
        print(f"Error inserting attendance data: {e}")
        return jsonify({"error": f"Failed to save attendance: {str(e)}"}), 500


@app.route('/download_attendance', methods=['GET'])
def download_attendance():
    """Generates and downloads today's attendance report in Excel format with lecture numbers."""
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch today's records from Supabase
    try:
        response = supabase.table('attendance').select('date, lecture_number, status, roll_no, students(name)').eq('date', current_date).order('roll_no', desc=False).order('lecture_number', desc=False).execute()
        
        if not response.data:
            return jsonify({"error": "No attendance records found for today."}), 404
        
        # Flatten the data and create date+lecture column
        data = []
        for row in response.data:
            # Create a combined column like "26-10-2025 (L1)"
            date_lecture = f"{row['date']} (L{row['lecture_number']})"
            data.append({
                'RollNo': row['roll_no'],
                'Name': row['students']['name'] if row.get('students') else 'N/A',
                'DateLecture': date_lecture,
                'Status': row['status']
            })

        df = pd.DataFrame(data)
        
        # Create pivot table: Rows = Students, Columns = Date+Lecture, Values = Status
        pivot_df = df.pivot_table(
            index=['RollNo', 'Name'], 
            columns='DateLecture', 
            values='Status', 
            aggfunc='first'
        ).reset_index()
        
        # Sort by RollNo
        pivot_df = pivot_df.sort_values('RollNo')
        
        # Generate filename
        filename = f"attendance_report_{current_date}.xlsx"
        
        # Create Excel file with formatting
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Format the header row
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)
            
            # Style header row
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # Add borders and alignment to all cells
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            # Color code the status cells
            present_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Light green
            absent_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")   # Light red
            
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=1, max_col=worksheet.max_column):
                for cell in row:
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                    
                    # Color code based on status
                    if cell.value == "Present":
                        cell.fill = present_fill
                        cell.font = Font(bold=True, color="006100")
                    elif cell.value == "Absent":
                        cell.fill = absent_fill
                        cell.font = Font(bold=True, color="9C0006")
            
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 10  # RollNo
            worksheet.column_dimensions['B'].width = 20  # Name
            for col in range(3, worksheet.max_column + 1):
                worksheet.column_dimensions[worksheet.cell(1, col).column_letter].width = 18  # Date+Lecture columns
        
        # Send the file
        return send_file(filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        print(f"Error generating report: {e}")
        return jsonify({"error": f"Failed to generate download report: {str(e)}"}), 500


@app.route('/download_total_attendance', methods=['GET'])
def download_total_attendance():
    """Generates and downloads complete attendance report for all students (all dates) in Excel format."""
    try:
        # Fetch ALL attendance records from Supabase
        response = supabase.table('attendance').select('date, lecture_number, status, roll_no, students(name)').order('roll_no', desc=False).order('date', desc=False).order('lecture_number', desc=False).execute()
        
        if not response.data:
            return jsonify({"error": "No attendance records found in the database."}), 404
        
        # Flatten the data and create date+lecture column
        data = []
        for row in response.data:
            # Create a combined column like "26-10-2025 (L1)"
            date_lecture = f"{row['date']} (L{row['lecture_number']})"
            data.append({
                'RollNo': row['roll_no'],
                'Name': row['students']['name'] if row.get('students') else 'N/A',
                'DateLecture': date_lecture,
                'Status': row['status']
            })

        df = pd.DataFrame(data)
        
        # Create pivot table: Rows = Students, Columns = Date+Lecture, Values = Status
        pivot_df = df.pivot_table(
            index=['RollNo', 'Name'], 
            columns='DateLecture', 
            values='Status', 
            aggfunc='first'
        ).reset_index()
        
        # Sort by RollNo
        pivot_df = pivot_df.sort_values('RollNo')
        
        # Generate filename with current timestamp
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"total_attendance_report_{current_timestamp}.xlsx"
        
        # Create Excel file with formatting
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            pivot_df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Format the header row
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF", size=11)
            
            # Style header row
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # Add borders and alignment to all cells
            thin_border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
            
            # Color code the status cells
            present_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Light green
            absent_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")   # Light red
            
            for row in worksheet.iter_rows(min_row=2, max_row=worksheet.max_row, min_col=1, max_col=worksheet.max_column):
                for cell in row:
                    cell.border = thin_border
                    cell.alignment = Alignment(horizontal='center', vertical='center')
                    
                    # Color code based on status
                    if cell.value == "Present":
                        cell.fill = present_fill
                        cell.font = Font(bold=True, color="006100")
                    elif cell.value == "Absent":
                        cell.fill = absent_fill
                        cell.font = Font(bold=True, color="9C0006")
            
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 10  # RollNo
            worksheet.column_dimensions['B'].width = 20  # Name
            for col in range(3, worksheet.max_column + 1):
                worksheet.column_dimensions[worksheet.cell(1, col).column_letter].width = 18  # Date+Lecture columns
        
        # Send the file
        return send_file(filename, as_attachment=True, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        print(f"Error generating total attendance report: {e}")
        return jsonify({"error": f"Failed to generate total attendance report: {str(e)}"})

    except Exception as e:
        print(f"Error generating total attendance report: {e}")
        return jsonify({"error": f"Failed to generate total attendance report: {str(e)}"}), 500


# --- Student Dashboard API (Search and Analytics) ---

@app.route('/api/search_student', methods=['GET'])
def search_student():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])

    # Use Supabase full-text search or simple filtering (Supabase handles this well)
    try:
        # Search by roll_no (exact match) or name (case-insensitive contains)
        # Supabase filtering example:
        response_roll = supabase.table('students').select('roll_no, name').eq('roll_no', query).limit(1).execute()
        if response_roll.data:
            return jsonify(response_roll.data)

        # Fallback to name search (requires PostgREST function or good RLS setup)
        # For simplicity, we assume we get a reasonable number of students and filter locally if needed, 
        # but the best practice is using Supabase's ILIKE or FTS.
        response_name = supabase.table('students').select('roll_no, name').execute() # Fetches all, inefficient but guaranteed to work
        results = [
            s for s in response_name.data 
            if query.lower() in s['name'].lower()
        ]
        return jsonify(results[:10]) # Limit to top 10 results

    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({"error": "Failed to search student"}), 500


@app.route('/api/student_dashboard/<int:roll_no>', methods=['GET'])
def student_dashboard(roll_no):
    """
    Provides attendance analytics for a single student.
    """
    today = datetime.now()
    
    # Calculate date ranges
    # 1. Current Month
    start_of_month = today.replace(day=1).strftime('%Y-%m-%d')
    end_of_month = today.strftime('%Y-%m-%d')

    # 2. Last Month (Simple approximation)
    last_month_start = (today.replace(day=1) - pd.DateOffset(months=1)).strftime('%Y-%m-%d')
    last_month_end = (today.replace(day=1) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    
    # 3. Overall
    
    try:
        # Step 1: Get Student Name
        student_response = supabase.table('students').select('name').eq('roll_no', roll_no).single().execute()
        student_name = student_response.data['name']
        
        # Step 2: Get ALL Attendance Records for the student
        attendance_response = supabase.table('attendance').select('date, status').eq('roll_no', roll_no).order('date', desc=True).execute()
        attendance_data = attendance_response.data
        
        if not attendance_data:
             return jsonify({
                "student_name": student_name,
                "monthly_stats": [],
                "overall_percentage": 0,
                "message": "No historical attendance data available."
            }), 200

        df_att = pd.DataFrame(attendance_data)
        df_att['date'] = pd.to_datetime(df_att['date'])
        
        def calculate_stats(df):
            total_days = len(df)
            present_days = len(df[df['status'] == 'Present'])
            percentage = round((present_days / total_days) * 100, 2) if total_days > 0 else 0
            return total_days, present_days, percentage

        # Overall Stats
        overall_total, overall_present, overall_percentage = calculate_stats(df_att)
        
        # Monthly Stats (Grouping by Month-Year)
        df_att['month_year'] = df_att['date'].dt.to_period('M')
        monthly_groups = df_att.groupby('month_year').apply(calculate_stats).reset_index()
        monthly_stats = []
        for index, row in monthly_groups.iterrows():
            total, present, percent = row[0]
            monthly_stats.append({
                "month": str(row['month_year']),
                "total_days": total,
                "present_days": present,
                "percentage": percent
            })
            
        return jsonify({
            "roll_no": roll_no,
            "student_name": student_name,
            "attendance_history": attendance_data,
            "overall_stats": {
                "total_days": overall_total,
                "present_days": overall_present,
                "percentage": overall_percentage
            },
            "monthly_stats": monthly_stats
        }), 200

    except Exception as e:
        print(f"Dashboard error: {e}")
        return jsonify({"error": f"Failed to retrieve student dashboard data: {e}"}), 500


if __name__ == '__main__':
    if load_model():
        # For production deployment (Render, Railway, etc.)
        # Set debug=False and host='0.0.0.0'
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
