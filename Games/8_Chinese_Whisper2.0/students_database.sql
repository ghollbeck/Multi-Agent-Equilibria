-- Students Database Schema
-- Create database for storing student information with 10 parameters
-- Most fields are binary (0/1) as requested

CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    speaks_english INTEGER CHECK (speaks_english IN (0, 1)),
    speaks_spanish INTEGER CHECK (speaks_spanish IN (0, 1)),
    grade_math_pass INTEGER CHECK (grade_math_pass IN (0, 1)),
    grade_science_pass INTEGER CHECK (grade_science_pass IN (0, 1)),
    grade_english_pass INTEGER CHECK (grade_english_pass IN (0, 1)),
    is_highschool INTEGER CHECK (is_highschool IN (0, 1)),
    is_active INTEGER CHECK (is_active IN (0, 1))
);

-- Create index on name for faster searches
CREATE INDEX IF NOT EXISTS idx_student_name ON students(last_name, first_name);

-- Create index on school level
CREATE INDEX IF NOT EXISTS idx_school_level ON students(is_highschool);

-- Sample insert statement (commented out - for reference)
-- INSERT INTO students (first_name, last_name, speaks_english, speaks_spanish, grade_math_pass, grade_science_pass, grade_english_pass, is_highschool, is_active)
-- VALUES ('John', 'Doe', 1, 0, 1, 1, 1, 1, 1); 