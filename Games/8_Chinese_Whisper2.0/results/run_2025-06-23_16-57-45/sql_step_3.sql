SELECT * FROM students 
WHERE is_highschool = 1 
AND grade_english_pass = 1 
AND (grade_math_pass = 0 OR grade_science_pass = 0) 
AND is_active = 1;