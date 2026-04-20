import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import seaborn as sns

class PCASoftware:
    def __init__(self):
        self.data = None
        self.reduced_data = None
        self.file_path = ""

    def load_file(self):
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        
        if self.file_path:
            try:
                if self.file_path.endswith('.csv'):
                    self.data = pd.read_csv(self.file_path, sep=None, engine='python')
                else:
                    self.data = pd.read_excel(self.file_path)
                messagebox.showinfo("Thành công", f"Đã nạp file: {self.file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc file: {e}")

    def preprocess_data(self):
        if self.data is None: return None

        # Tìm cột nhãn -> Phân loại
        potential_labels = self.data.select_dtypes(exclude=[np.number])
    
        if not potential_labels.empty:
            self.labels = potential_labels.iloc[:, 0]
        else:
            for col in self.data.columns:
                if self.data[col].nunique() < 15: 
                    self.labels = self.data[col]
                    break

        # Số hóa dữ liệu (Chỉ giữ các cột số)
        numeric_df = self.data.select_dtypes(include=[np.number])
        
        # Loại bỏ cột toàn giá trị trống
        numeric_df = numeric_df.dropna(axis=1, how='all')

        # Loại bỏ các cột định danh (ID) - các cột có giá trị không trùng lặp
        for col in numeric_df.columns:
            if numeric_df[col].nunique() == len(numeric_df):
                numeric_df = numeric_df.drop(columns=[col])

        # Xử lý giá trị trống (Điền bằng trung bình cột)
        numeric_df = numeric_df.fillna(numeric_df.mean())
        return numeric_df

    def run_pca(self, n_components=2):
        X = self.preprocess_data()
        if X is None: return
        
        # 1. Chuẩn hóa (Standardization)
        X_std = (X - X.mean()) / X.std()
        
        # 2. Ma trận hiệp phương sai
        cov_matrix = np.cov(X_std.T)

        # Kiểm tra nếu có giá trị NaN hoặc Inf thì thay bằng 0
        cov_matrix = np.nan_to_num(cov_matrix)
        
        # 3. Trị riêng & Vectơ riêng
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        
        # 4. Sắp xếp và chọn thành phần
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        
        # Lấy k vectơ riêng đầu tiên
        self.components = eigen_vectors[:, :n_components]
        
        # 5. Chiếu dữ liệu (Giảm chiều)
        self.reduced_data = np.dot(X_std, self.components)
        
        # Tính toán sai số/đánh giá
        total_variance = np.sum(eigen_values)
        explained_variance = eigen_values[:n_components] / total_variance
        cumulative_variance = np.cumsum(explained_variance)
        
        self.show_results(explained_variance, cumulative_variance)

    def show_results(self, exp_var, cum_var):
       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Biểu đồ sai số (Scree Plot)
        ax1.bar(range(1, len(exp_var)+1), exp_var, label='Phương sai thành phần')
        ax1.plot(range(1, len(exp_var)+1), cum_var, marker='o', color='red', label='Tích lũy')
        ax1.set_title("Đánh giá độ bảo toàn thông tin")
        ax1.set_xlabel("Số lượng PC")
        ax1.legend()

        # Biểu đồ không gian mới (2D)
        if hasattr(self, 'labels') and self.labels is not None:
            sns.scatterplot(
                x=self.reduced_data[:, 0], 
                y=self.reduced_data[:, 1], 
                hue=self.labels,       
                palette='Set1',     
                ax=ax2,                 
                alpha=0.7, 
                s=60
            )
            ax2.legend(title='Nhóm dữ liệu', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax2.scatter(self.reduced_data[:, 0], self.reduced_data[:, 1], alpha=0.6, c='teal')
    
        ax2.set_title(f"Dữ liệu sau giảm chiều (Giữ lại {cum_var[-1]:.2%})")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")
        
        plt.tight_layout()
        plt.savefig("result_pca.png")
        plt.show()
        
        
        res_df = pd.DataFrame(self.reduced_data, columns=[f'PC_{i+1}' for i in range(self.reduced_data.shape[1])])
        res_df.to_csv("data_reduced.csv", index=False)
        print("Đã lưu kết quả vào file 'data_reduced.csv' và ảnh 'result_pca.png'")


if __name__ == "__main__":
    app = PCASoftware()
    print("--- PHẦN MỀM GIẢM CHIỀU DỮ LIỆU PCA ---")
    app.load_file()
    if app.data is not None:
        k = int(input("Số chiều muốn giảm?: ")) #Thường là giảm 2 chiều nha!
        app.run_pca(k)