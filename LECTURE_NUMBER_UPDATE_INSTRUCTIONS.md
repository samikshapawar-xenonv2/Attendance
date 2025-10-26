# Database Update Instructions - Lecture Number Feature

## Overview
This update adds support for multiple attendance sessions per day (Lecture 1, 2, 3, etc.)

## Step 1: Update Supabase Database Schema

You need to add a `lecture_number` column to the `attendance` table.

### Option A: Using Supabase SQL Editor (Recommended)

1. Go to your Supabase Dashboard: https://supabase.com/dashboard
2. Select your project
3. Click on "SQL Editor" in the left sidebar
4. Click "New Query"
5. Copy and paste the following SQL:

```sql
-- Add lecture_number column to attendance table
ALTER TABLE public.attendance 
ADD COLUMN lecture_number INT NOT NULL DEFAULT 1;

-- Update the index for better performance
DROP INDEX IF EXISTS idx_attendance_roll_date;
CREATE INDEX idx_attendance_roll_date_lecture ON public.attendance (roll_no, date, lecture_number);

-- Add a comment to document the column
COMMENT ON COLUMN public.attendance.lecture_number IS 'Lecture number for the day (1, 2, 3, etc.). Auto-incremented per date.';
```

6. Click "RUN" button
7. You should see "Success. No rows returned"

### Option B: Using Table Editor

1. Go to Supabase Dashboard → Table Editor
2. Select `attendance` table
3. Click the "+" button to add a new column
4. Configure:
   - Name: `lecture_number`
   - Type: `int4`
   - Default value: `1`
   - NOT NULL: ✓ (checked)
5. Click "Save"

## Step 2: Restart Your Flask Application

After updating the database schema, restart your Flask app:

```powershell
# Stop the current Flask app (Ctrl+C in the terminal)
# Then restart:
python app.py
```

## How It Works

### Auto-Detection of Lecture Number
- When you click "Finalize & Save Attendance", the system automatically detects which lecture it is for today
- If this is the first attendance of the day → Lecture 1
- If you already took Lecture 1 today → Next one will be Lecture 2
- And so on...

### Example Workflow
**October 26, 2025:**
- 9:00 AM - Take attendance → Saved as "Lecture 1 on 26-10-2025"
- 11:00 AM - Take attendance → Saved as "Lecture 2 on 26-10-2025"
- 2:00 PM - Take attendance → Saved as "Lecture 3 on 26-10-2025"

**October 27, 2025:**
- 9:00 AM - Take attendance → Saved as "Lecture 1 on 27-10-2025"
(Lecture counter resets for each new day)

### Excel Export Format
When you download the total attendance report, columns will show:
- `26-10-2025 (L1)` - Lecture 1 on Oct 26
- `26-10-2025 (L2)` - Lecture 2 on Oct 26
- `26-10-2025 (L3)` - Lecture 3 on Oct 26
- `27-10-2025 (L1)` - Lecture 1 on Oct 27
- And so on...

## Testing

1. Go to Live Attendance page
2. Take attendance and click "Finalize & Save"
3. You should see: "Attendance for Lecture 1 finalized..."
4. Take attendance again (same day)
5. Click "Finalize & Save"
6. You should see: "Attendance for Lecture 2 finalized..."
7. Download Total Attendance to see the Excel with lecture numbers

## Troubleshooting

**Error: "column 'lecture_number' does not exist"**
- Solution: Run the SQL script in Supabase SQL Editor

**Error: "null value in column 'lecture_number'"**
- Solution: Make sure the DEFAULT value is set to 1 in the column definition

**Old attendance records showing as NULL**
- This is expected for records created before the update
- They will default to lecture_number = 1
- New records will have proper lecture numbers

## Files Modified
- `app.py` - Updated finalize_attendance() and download_total_attendance()
- `update_database_schema.sql` - SQL script for database update
