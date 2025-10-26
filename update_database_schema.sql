-- SQL Script to Update Attendance Table with Lecture Number Support
-- Run this in your Supabase SQL Editor

-- Step 1: Add lecture_number column to attendance table
ALTER TABLE public.attendance 
ADD COLUMN lecture_number INT NOT NULL DEFAULT 1;

-- Step 2: Update the index to include lecture_number for better performance
DROP INDEX IF EXISTS idx_attendance_roll_date;
CREATE INDEX idx_attendance_roll_date_lecture ON public.attendance (roll_no, date, lecture_number);

-- Step 3: Add a comment to document the column
COMMENT ON COLUMN public.attendance.lecture_number IS 'Lecture number for the day (1, 2, 3, etc.). Auto-incremented per date.';

-- Optional: View the updated table structure
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_name = 'attendance' AND table_schema = 'public'
ORDER BY ordinal_position;
