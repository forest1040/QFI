# from float import float

print_matrix = lambda matrix: print(*matrix, sep="\n")  # 行列表示用


def lu_decomposition(
    A: list[list[float]], N
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    正方行列AをLU分解する。

    Parameters
    ----------
    A : list[list[float]]
        LU分解する正方行列A。

    Returns
    -------
    L : list[list[float]]
        下三角行列L。
    U : list[list[float]]
        上三角行列U。
    P : list[list[float]]
        ピボット選択の置換行列P。
    """
    L: list[list[float]] = [
        [float(1) if i == j else float(0) for j in range(N)] for i in range(N)
    ]  # 単位行列
    U: list[list[float]] = [[float(0) for j in range(N)] for i in range(N)]  # 零行列
    P: list[list[float]] = [
        [float(1) if i == j else float(0) for j in range(N)] for i in range(N)
    ]  # 単位行列

    # ピボット選択
    for k in range(N):
        abs_col: list[float] = [abs(A[i][k]) for i in range(N)]
        max_index: int = abs_col.index(max(abs_col))  # 絶対値が最も大きい成分を探す

        # swap
        A[k], A[max_index] = A[max_index], A[k]
        P[k], P[max_index] = P[max_index], P[k]

    for k in range(N):
        # U
        for j in range(k, N):
            U[k][j] = A[k][j]
            for s in range(k):
                U[k][j] -= L[k][s] * U[s][j]

        # L
        for i in range(1 + k, N):
            L[i][k] = A[i][k]
            for s in range(k):
                L[i][k] -= L[i][s] * U[s][k]
            if U[k][k] is not None and U[k][k] != 0:
                # print("U[k][k]:", U[k][k])
                L[i][k] /= U[k][k]

    return L, U, P


def backward_substitution(A: list[list[float]], B: list[float], N) -> list[float]:
    """
    後退代入で方程式AX = Bを解く。

    Parameters
    ----------
    A : list[list[float]]
        上三角行列A。
    B : list[float]
        行列B。

    Returns
    -------
    X : list[float]
        方程式の解。
    """
    X: list[float] = [float(0) for i in range(N)]
    for i in reversed(range(0, N)):
        X[i] = B[i]
        for k in range(i + 1, N):
            X[i] -= A[i][k] * X[k]

        if A[i][i] is not None and A[i][i] != 0:
            X[i] /= A[i][i]

    return X


def multiply_permutation_matrix(P: list[list[float]], B: list[float], N) -> list[float]:
    """
    置換行列Pを行列Bに左からかける。

    Parameters
    ----------
    P : list[list[float]]
        置換行列P。
    B : list[float]
        行列B。

    Returns
    -------
    PB : list[float]
        置換後の行列。
    """
    PB: list[float] = [float(0) for i in range(N)]
    for i in range(N):
        for j in range(N):
            PB[i] += P[i][j] * B[j]

    return PB


if __name__ == "__main__":
    # 標準入力
    N: int = int(input())
    A: list[list[float]] = [list(map(float, input().split())) for i in range(N)]
    B: list[float] = list(map(float, input().split()))

    # LU分解
    L: list[list[float]]
    U: list[list[float]]
    P: list[list[float]]
    L, U, P = lu_decomposition(A)

    # LY = PBを解く
    L = [row[::-1] for row in L[:]][::-1]  # 後退代入と上下逆なので逆順に
    PB: list[float] = multiply_permutation_matrix(P, B)[::-1]  # 置換行列をかけて後退代入と上下逆なので逆順に
    Y: list[float] = backward_substitution(L, PB)[::-1]

    # UX = Yを解く
    X: list[float] = backward_substitution(U, Y)

    # 標準出力
    print(*X, sep="\n")
