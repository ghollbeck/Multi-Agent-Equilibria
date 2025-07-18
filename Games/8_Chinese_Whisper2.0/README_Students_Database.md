# Students Database

This folder contains a SQLite database system for managing student information with 10 parameters per student.

## Database Structure

The `students` table contains the following 10 columns (most are binary 0/1 fields):

1. **student_id** (INTEGER, PRIMARY KEY, AUTO INCREMENT) - Unique identifier for each student
2. **first_name** (VARCHAR(50), NOT NULL) - Student's first name
3. **last_name** (VARCHAR(50), NOT NULL) - Student's last name
4. **speaks_english** (INTEGER, 0/1) - Whether student speaks English (1=Yes, 0=No)
5. **speaks_spanish** (INTEGER, 0/1) - Whether student speaks Spanish (1=Yes, 0=No)
6. **grade_math_pass** (INTEGER, 0/1) - Math grade pass/fail (1=Pass, 0=Fail)
7. **grade_science_pass** (INTEGER, 0/1) - Science grade pass/fail (1=Pass, 0=Fail)
8. **grade_english_pass** (INTEGER, 0/1) - English grade pass/fail (1=Pass, 0=Fail)
9. **is_highschool** (INTEGER, 0/1) - School level (1=High School, 0=Primary School)
10. **is_active** (INTEGER, 0/1) - Student status (1=Active, 0=Inactive)

## Files

- `students_database.sql` - SQL schema definition
- `students_db_manager.py` - Python class for database operations
- `students.db` - SQLite database file (created when first run)

## Usage

### Initialize Database
```python
from students_db_manager import StudentsDBManager

# Create database manager (creates database if it doesn't exist)
db = StudentsDBManager()
```

### Add a Student
```python
db.add_student(
    first_name="John",
    last_name="Doe", 
    speaks_english=1,      # Speaks English
    speaks_spanish=0,      # Doesn't speak Spanish
    grade_math_pass=1,     # Passed Math
    grade_science_pass=1,  # Passed Science
    grade_english_pass=1,  # Passed English
    is_highschool=1,       # High School student
    is_active=1            # Active student
)
```

### Retrieve Students
```python
# Get all students
all_students = db.get_all_students()

# Get student by ID
student = db.get_student_by_id(1)

# Search students
results = db.search_students("John")
```

### Update Student
```python
db.update_student(1, grade_math_pass=0, is_active=0)  # Failed math, set to inactive
```

### Delete Student
```python
db.delete_student(1)
```

### Get Statistics
```python
count = db.get_student_count()
print(f"Total students: {count}")
```

## Running the Example

To test the database with sample data:

```bash
cd Games/8_Chinese_Whisper2.0
python students_db_manager.py
```

This will create the database, add 5 sample students, and display them with detailed information.

## Database Features

- **Binary Fields**: Most fields use 0/1 values for efficient storage and querying
- **Data Validation**: All binary fields restricted to 0 or 1 values
- **Language Support**: Track English and Spanish speaking capabilities
- **Grade Tracking**: Pass/Fail status for Math, Science, and English
- **School Level**: Distinguish between High School and Primary School students
- **Student Status**: Track active/inactive students
- **Indexes**: Optimized for name and school level searches
- **Error Handling**: Comprehensive error handling for all operations
- **Flexible Updates**: Update any combination of fields

## Requirements

- Python 3.6+
- SQLite3 (included with Python)
- No external dependencies required 