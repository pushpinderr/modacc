import matplotlib.pyplot as plt
labels = ['Python', 'C++', 'Ruby']
sizes = [100,100,100]
# Plot
plt.pie(sizes, labels=labels, 
        autopct='%1.2f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()