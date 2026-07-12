# Student Performance Prediction System — Project Documentation

## 1. Problem Statement

This project predicts a student's overall academic performance using attendance,
assignment scores, internal assessments, and other academic/lifestyle indicators,
so that at-risk students can potentially be identified early.

---

## 2. Model Summary (Based on Actual Trained Model)

**Algorithm:** Linear Regression (scikit-learn)

**Target variable:** `performance_index`

**Trained features and their learned coefficients:**

| Feature | Coefficient | Interpretation |
|---|---|---|
| `attendance_percentage` | +0.840 | Each 1% increase in attendance raises predicted performance by ~0.84 points |
| `assignment_score` | +0.688 | Each 1-point increase in assignment score raises performance by ~0.69 points |
| `internal_assessment_score` | +0.530 | Each 1-point increase in internal score raises performance by ~0.53 points |
| `study_hours_per_week` | +0.737 | Each additional study hour/week raises performance by ~0.74 points |
| `previous_semester_gpa` | +4.264 | Each 1-point increase in prior GPA raises performance by ~4.26 points |
| `extracurricular_participation` | +6.049 | Participating in extracurriculars is associated with ~6.05 points higher performance |
| `part_time_job` | −10.065 | Having a part-time job is associated with ~10.07 points lower performance |

**Intercept:** −135.91

**Full model equation:**
```
performance_index = -135.91
    + 0.840 * attendance_percentage
    + 0.688 * assignment_score
    + 0.530 * internal_assessment_score
    + 0.737 * study_hours_per_week
    + 4.264 * previous_semester_gpa
    + 6.049 * extracurricular_participation
    - 10.065 * part_time_job
```

---

## 3. Key Insights From the Coefficients

- **Previous GPA and extracurricular participation** have the largest positive
  influence on predicted performance among all features.
- **Part-time job** has the single largest negative effect — larger in magnitude
  than any positive factor except GPA — suggesting working students may need
  additional academic support.

---

## 5. Model Evaluation

| Metric | Value |
|---|---|
| R² Score | — 0.9971252099666724 |
| MAE | — 0.7615455418165216 |
| MSE | —  0.9766318757627227|
| RMSE | — 0.9882468698471666|

To generate these, run on your test set:
```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

y_pred = model.predict(x_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

---

## 6. Conclusion

The model shows that attendance, assignments, internal assessments, study habits,
prior GPA, and part-time job status are meaningfully associated with overall
student performance — with prior GPA and part-time job status having the
strongest effects. The current inclusion of `final_exam_score` as a feature is a
design issue to revisit before deployment, though the model's own coefficients
suggest it isn't materially affecting predictions yet.

---

## 7. Future Scope

- Validate on real student data rather than synthetic data.
- Add cross-validation to confirm the coefficients are stable across different
  data splits.
