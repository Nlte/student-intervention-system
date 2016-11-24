import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import pandas as pd

student_data = pd.read_csv("student-data.csv")
print student_data.head()
student_data.plot.scatter(x='passed', y='internet')
#x = student_data[]
#plt.scatter(x, y)
#plt.show()
