def matrix_rank(matrix):
    import copy
    mat = copy.deepcopy(matrix)
    m = len(mat)
    n = len(mat[0]) if m > 0 else 0
    rank = 0
    for r in range(n):
        pivot_row = None
        for i in range(rank, m):
            if abs(mat[i][r]) > 1e-12:
                pivot_row = i
                break
        if pivot_row is not None:
            mat[rank], mat[pivot_row] = mat[pivot_row], mat[rank]
            for i in range(rank + 1, m):
                if abs(mat[i][r]) > 1e-12:
                    factor = mat[i][r] / mat[rank][r]
                    for j in range(r, n):
                        mat[i][j] -= factor * mat[rank][j]
            rank += 1
    return rank

def determinant(matrix):
    n = len(matrix)
    mat = [row[:] for row in matrix]  # 深拷贝
    det = 1
    for i in range(n):
        # 找到当前列主元素
        max_row = i
        for j in range(i+1, n):
            if abs(mat[j][i]) > abs(mat[max_row][i]):
                max_row = j
        if abs(mat[max_row][i]) < 1e-12:
            return 0
        if max_row != i:
            mat[i], mat[max_row] = mat[max_row], mat[i]
            det *= -1
        det *= mat[i][i]
        for j in range(i+1, n):
            factor = mat[j][i] / mat[i][i]
            for k in range(i, n):
                mat[j][k] -= factor * mat[i][k]
    return det

def main():
    try:
        cols = int(input("请输入矩阵的列数: ").strip())
        rows = int(input("请输入矩阵的行数: ").strip())
        if cols <= 0 or rows <= 0:
            print("列数和行数必须为正整数！")
            return
        values = input(f"请输入{rows * cols}个矩阵元素，用空格隔开:\n").strip().split()
        if len(values) != rows * cols:
            print("输入元素数量与矩阵大小不符！")
            return
        try:
            numbers = list(map(float, values))
        except ValueError:
            print("所有元素必须为数字！")
            return
        matrix = []
        for i in range(rows):
            matrix.append(numbers[i*cols:(i+1)*cols])
        rank = matrix_rank(matrix)
        print(f"该矩阵的秩为: {rank}")
        if rank == min(rows, cols):
            print("满秩！")
        # 如为方阵，求行列式
        if rows == cols:
            det = determinant(matrix)
            print(f"方阵的行列式为: {det}")
    except Exception as e:
        print("发生错误:", e)

if __name__ == "__main__":
    main()