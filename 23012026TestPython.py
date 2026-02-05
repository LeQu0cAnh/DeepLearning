# Hiển thị hành ảnhn bằng OpenCV 
# import os
# import cv2
# import matplotlib.pyplot as plt

# def main():
#     # 1. Thêm chữ 'r' phía trước để tránh lỗi Unicode (\U)
#     # 2. Đảm bảo đường dẫn thư mục là chính xác
#     path = r'C:\Users\lequo\Things\pictures'
    
#     # Kiểm tra xem thư mục có tồn tại không
#     if not os.path.exists(path):
#         print(f"Lỗi: Không tìm thấy đường dẫn: {path}")
#         return

#     # 3. Sử dụng os.path.join để tự động xử lý dấu gạch chéo đúng cách
#     img_name = 'logo.png'
#     img_path = os.path.join(path, img_name)
    
#     # Đọc ảnh
#     img = cv2.imread(img_path)

#     # Kiểm tra xem ảnh có đọc được không
#     if img is not None:
#         # HIỂN THỊ BẰNG OPENCV 
#         cv2.imshow('Hinh anh cua toi', img)
#         cv2.waitKey(0) # Dừng màn hình để xem, nhấn phím bất kỳ để đóng
#         cv2.destroyAllWindows()
        
#         # HIỂN THỊ BẰNG MATPLOTLIB (Vẽ đồ thị)
#         # Chuyển BGR sang RGB để hiện đúng màu
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.imshow(img_rgb)
#         plt.title("Hien thi qua Matplotlib")
#         plt.show()
#     else:
#         print(f"Không thể đọc được file ảnh tại: {img_path}")
#         print("Vui lòng kiểm tra lại tên file '1.jpg' có trong thư mục không.")

# if __name__ == "__main__":
#     main()


#================================================================================


# # tạo hình chứa đầu vào là tọa độ x,y dựa vào thư viện matplotlib
# import matplotlib.pyplot as plt
# x = [1, 2, 3, 4, 5]
# y = [10, 16, 20, 30, 25]
# # vẽ đồ thị với màu đỏ
# #plt.plot(x,y, color='red')
# #plt.plot(x,y) # màu mặc định : blue
# # thêm tiêu đề và nhãn trục
# plt.title('Vi du ve matplotlib')
# plt.xlabel('Truc x')
# plt.ylabel('Truc y')
# plt.show()
#================================================================================
# # Vẽ biểu đồ: Line Chart
# import matplotlib.pyplot as plt
# # dữ liệu 1
# x = [1, 2, 3, 4, 5]
# y = [10, 16, 20, 30, 25]
# #dữ liệu 2
# x1 = [1, 2, 3, 4, 5]
# y1 = [20, 15, 8, 9, 20]

# plt.plot(x, y, 'bo-', label='Dữ liệu 1') # Vẽ đường 1
# plt.plot(x1, y1, marker='o', color='r', linestyle='-', label='Dữ liệu 2')# Vẽ đường 2
# plt.legend() # Hiển thị chú giải
# plt.xlabel('Trục x') # Đặt nhãn cho trục x
# plt.ylabel('Trục y') # Đặt nhãn cho trục y
# plt.show() # Hiển thị biểu đồ


#================================================================================

# Vẽ biểu đồ cột Bar Chart
# import matplotlib.pyplot as plt

# categories = ['A', 'B', 'C', 'D', 'E']
# values = [15, 10, 25, 12, 18]

# plt.bar(categories, values, color='g', alpha=0.6) # Vẽ biểu đồ cột
# plt.xlabel('Danh mục') # Đặt nhãn cho trục x
# plt.ylabel('Giá trị') # Đặt nhãn cho trục y
# plt.show() # Hiển thị biểu đồ

#================================================================================# 
# Vẽ biểu đồ hình tròn Pie Chart
# import matplotlib.pyplot as plt
# my_labels = ['Anh', 'Binh', 'Chau', 'Dai']
# sizes = [15, 30, 45, 10] # Kích thước từng phần, cái này là giá trị, máy tự quy đổi thành tỉ lệ %
# my_colors = ['red', 'green', 'blue', 'yellow']

# plt.pie(sizes, labels=my_labels, colors = my_colors, autopct='%1.1f%%') # Vẽ biểu đồ hình tròn
# plt.title('Biểu đồ hình tròn')
# plt.show()



#================================================================================

# import matplotlib.pyplot as plt
# #note về biểu đồ hộp Box Plot: Q1: cạnh dưới, 25% gái trị nằm ở dưới Q2: trung vị, chia đôi dữ liệu Q3: cạnh trên, 75% giá trị nằm dưới
# #Độ dài của hộp đại diện cho Khoảng cách tứ phân vị (IQR), nơi chứa 50% lượng dữ liệu tập trung nhất.
# # Nhìn vào Q2 để biết dữ liệu lệch về bên nào, nhìn vào IQR để biết dữ liệu phân tán hay tập trung
# data = [15, 18, 22, 30, 35, 45, 50, 55, 65]

# plt.boxplot(data)  # Vẽ biểu đồ hộp
# plt.title('Biểu đồ hộp ví dụ')  # Đặt tiêu đề cho biểu đồ
# plt.ylabel('Giá trị')  # Đặt nhãn cho trục y
# plt.show()  # Hiển thị biểu đồ
#================================================================================

# import matplotlib.pyplot as plt
# # note về biểu đồ violin Violin Plot: Kết hợp giữa biểu đồ hộp và biểu đồ mật độ, giúp hiển thị phân phối dữ liệu một cách trực quan hơn.
# # Hình dạng của "violin" thể hiện mật độ dữ liệu tại các mức giá trị khác nhau: phần rộng hơn cho thấy mật độ dữ liệu cao hơn, trong khi phần hẹp hơn cho thấy mật độ thấp hơn.
# data = [15, 18, 22, 30, 35, 45, 50, 55, 65]

# plt.violinplot(data)  # Vẽ biểu đồ violin
# plt.title('Biểu đồ violin ví dụ')  # Đặt tiêu đề cho biểu đồ
# plt.ylabel('Giá trị')  # Đặt nhãn cho trục y
# plt.show()  # Hiển thị biểu đồ
     
     
#================================================================================
     
#Giới thiệu về Numpy
import numpy as np

# #Mảng trong Numpy, bắt đầu từ 0
# # Tạo mảng
# a = np.array([50,75,300])
# # In mảng
# print(a)
# # In ra 1 phần tử dựa vào stt của phần tử đó trong mảng
# element = a[1]
# print(element)
# # Tạo mảng 2 chiều
# matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print("Matrix:")
# print(matrix)
# # In một phần tử dựa vào vị trí hàng và cột 
# element = matrix[1, 2]
# print(element)

#================================================================================
# Cách tạo mảng
# Tạo mảng toàn là số 0
# a = np.zeros(5)
# print(a)
# Tạo mảng toàn là số 1
# a = np.ones(5)
# print(a)
# Tạo mảng
# a = np.empty(5) # giá trị trong mảng có thể là rác
# print(a)
# Tạo mảng từ 0 đến số n-1
# a = np.arange(100)
# print(a)
# Tạo mảng gồm các phần tử với khoảng cách đều nhau
# a = np.linspace(0, 10, num=5) # Tạo mảng từ a đến b gồm x phần tử
# print(a)
# Xác định kiểu dữ liệu
# a = np.ones(5, dtype=np.int64)
# a



#================================================================================

# Thêm, xóa, sắp xếp mảng 1 chiều
# Tạo mảng ban đầu
# arr = np.array([7, 1, 3, 6, 2, 4, 5])
# print(arr)
# Sắp xếp
# arr = np.sort(arr)
# print(arr)
# Sắp xếp ngược
# arr = np.sort(arr)[::-1]
# print(arr)
# Thêm phần tử vào mảng
# arr = np.append(arr, 100)
# print(arr)
# Xóa đi một ví trí nào đó trong mảng
# arr = np.delete(arr, 3)
# print(arr)

#sắp xếp trên mảng 2 chiều
# arr = np.array([[3, 1, 2],
#                 [4, 6, 8],
#                 [9, 7, 5]])
# print(arr)
# sap_xep_theo_hang = np.sort(arr, axis=1) # tăng dần
# print(sap_xep_theo_hang)
# sap_xep_theo_cot = np.sort(arr, axis=0) # tăng dần
# print(sap_xep_theo_cot)
# sap_xep_theo_hang_giam = np.sort(arr, axis=1)[::-1] # giảm dần
# print(sap_xep_theo_hang_giam)

#================================================================================

# Tạo một mảng 2 chiều
# arr = np.array([[1, 2, 3], [4, 5, 6]])

# # Sử dụng các thuộc tính để lấy thông tin về mảng
# so_chieu = arr.ndim  # Số chiều (2 chiều)
# kich_thuoc = arr.size  # Kích thước (tổng số phần tử, 6)
# hinh_dang = arr.shape  # Hình dạng (số hàng x số cột, (2, 3))

# print("Số chiều:", so_chieu)
# print("Kích thước:", kich_thuoc)
# print("Hình dạng:", hinh_dang)

# Chuyển đổi kiểu dữ liệu
# arr = np.array([1, 2, 3, 4, 5])
# arr_float = arr.astype(float)
# print(arr_float)
     

# Thay đổi hình dạng của mảng
# arr = np.array([1, 2, 3, 4, 5, 6])
# reshaped_arr = arr.reshape(2,3)
# print(reshaped_arr)
# reshaped_arr = arr.reshape(3,2)
# print(reshaped_arr)
     

# Cắt lát mảng
# arr = np.array([1, 2, 3, 4, 5])

# sub_arr = arr[1:4]  # Cắt từ phần tử thứ 1 đến 4
# print(sub_arr) # 2 3 4 5

# sub_arr = arr[:-1]  # 1 2 3 4
# print(sub_arr)

# sub_arr = arr[-2:]  # 4 5
# print(sub_arr)
     

# Chuyển vị mảng
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr)
# Để chuyển vị mảng, chúng ta có thể sử dụng .T hoặc hàm numpy.transpose().
# transposed_arr = arr.T
# print(transposed_arr)
     

# Nối mảng
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# arr = np.concatenate((arr1, arr2))
# print(arr)
     

# - Hàm numpy.sum() cho phép tính tổng các phần tử trong mảng.
# arr = np.array([1, 2, 3, 4, 5])
# total = np.sum(arr)
# print("Tổng của mảng arr:", total)

# III. Hàm tính trung bình (Mean)
# - Hàm numpy.mean() tính giá trị trung bình của các phần tử trong mảng.
# average = np.mean(arr)
# print("Giá trị trung bình của mảng arr:", average)

# IV. Hàm tìm giá trị lớn nhất và nhỏ nhất (Max và Min)
# - Hàm numpy.max() và numpy.min() dùng để tìm giá trị lớn nhất và nhỏ nhất trong mảng.
# max_value = np.max(arr)
# min_value = np.min(arr)
# print("Giá trị lớn nhất trong mảng arr:", max_value)
# print("Giá trị nhỏ nhất trong mảng arr:", min_value)

# V. Hàm tính độ lệch chuẩn (Standard Deviation)
# - Hàm numpy.std() tính độ lệch chuẩn của mảng, đo lường mức độ phân tán của dữ liệu.
# std_deviation = np.std(arr)
# print("Độ lệch chuẩn của mảng arr:", std_deviation)

# VI. Hàm tính phương sai (Variance)
# - Hàm numpy.var() tính phương sai của mảng, đo lường mức độ biến thiên của dữ liệu.
# variance = np.var(arr)
# print("Phương sai của mảng arr:", variance)

# VII. Hàm tính tổng tích chập (Dot Product)
# - Hàm numpy.dot() tính tổng tích chập của hai mảng (vector).
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# dot_product = np.dot(arr1, arr2)
# print("Tổng tích chập của arr1 và arr2:", dot_product)

# from scipy import linalg
     

# Giải hệ phương trình tuyến tính
# Định nghĩa hệ phương trình tuyến tính
# A = np.array([[2, 1],[3, 2]])
# b = np.array([5,7])

# Giải hệ phương trình tuyến tính
# x = linalg.solve(A, b)
# print("Kết quả: ", x)
     

# from scipy import integrate

# Tính tích phân của một hàm số
# Định nghĩa hàm f(x) = x^2
# def my_function(x):
#     return x**2
# Tính tích phân của f(x) từ 0 đến 1
# integral = integrate.quad(my_function, 0, 1);
# print("Tích phân f(x) từ 0 đến 1: ", integral)
     

# Tính giá trị riêng và vector riêng của một ma trận:
# A = np.array([[2,1], [3,2]])

# Tính giá trị riêng và vector riêng của ma trận
# evals, evecs = linalg.eig(A)

# print("Giá trị riêng: ", evals)
# print("Vector riêng: ", evecs)