import math

def point_between(a, b, r):
    # Tính khoảng cách giữa a và b
    d = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    
    # Tính tọa độ của điểm c
    cx = a[0] + r * (b[0] - a[0]) / d
    cy = a[1] + r * (b[1] - a[1]) / d
    
    return cx, cy

# Tọa độ điểm A và B
a = (1, 1)
b = (4, 5)

# Khoảng cách từ A đến C
r = 1

# Tìm vị trí của điểm C
c = point_between(a, b, r)
print(c)