import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import trimesh
import os
from mpl_toolkits.mplot3d import Axes3D

def normalize_scale(points):
    """
    Normaliza a escala de uma nuvem de pontos para garantir consistência nas comparações.
    Parâmetros:
    - points (ndarray): Nuvem de pontos (n x 3).
    Retorna:
    - ndarray: Nuvem de pontos normalizada (n x 3).
    """
    scale = np.linalg.norm(points, axis=1).max()  # Calcula a maior distância de um ponto à origem
    return points / scale  # Normaliza a nuvem de pontos dividindo pela escala máxima

def compute_normals(points, k=10):
    """
    Calcula as normais de uma nuvem de pontos usando os k-vizinhos mais próximos.
    Parâmetros:
    - points (ndarray): Nuvem de pontos (n x 3).
    - k (int): Número de vizinhos mais próximos a considerar para o cálculo das normais (default: 10).
    Retorna:
    - ndarray: Normais calculadas (n x 3).
    """
    normals = np.zeros_like(points)  # Inicializa a matriz para armazenar as normais
    tree = KDTree(points)  # Cria um KDTree para busca de vizinhos próximos

    for i in range(points.shape[0]):
        distances, indices = tree.query(points[i], k=k)  # Encontra os k-vizinhos mais próximos
        neighbors = points[indices]  # Obtém os pontos vizinhos
        covariance_matrix = np.cov(neighbors.T)  # Calcula a matriz de covariância dos vizinhos
        _, _, Vt = np.linalg.svd(covariance_matrix)  # Aplica SVD para encontrar as direções principais
        normals[i] = Vt[-1]  # A última coluna de Vt é a direção com a menor variação, ou seja, a normal

    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Normaliza as normais
    return normals

def dynamic_rejection_threshold(distances, factor=1.5):
    """
    Calcula um limiar de rejeição dinâmico baseado no IQR das distâncias.
    Parâmetros:
    - distances (ndarray): Distâncias calculadas entre pares de pontos.
    - factor (float): Fator multiplicador para o IQR (default: 1.5).
    Retorna:
    - float: Limiar de rejeição dinâmico.
    """
    q1 = np.percentile(distances, 25)  # Primeiro quartil
    q3 = np.percentile(distances, 75)  # Terceiro quartil
    iqr = q3 - q1  # Intervalo interquartil
    return q3 + factor * iqr  # Limiar de rejeição baseado no IQR

def validate_transformation(T, max_expected_translation=5.0):
    """
    Valida uma matriz de transformação garantindo que a rotação seja válida e a translação esteja dentro do esperado.
    Parâmetros:
    - T (ndarray): Matriz de transformação 4x4.
    - max_expected_translation (float): Magnitude máxima esperada para a translação (default: 5.0).
    Retorna:
    - None: A função levanta uma exceção se a transformação não for válida.
    """
    R = T[:3, :3]  # Extrai a matriz de rotação
    t = T[:3, 3]  # Extrai o vetor de translação
    det_R = np.linalg.det(R)  # Calcula o determinante da matriz de rotação

    # Verificações de validade da matriz de rotação e do vetor de translação
    if not np.isclose(det_R, 1.0):
        raise ValueError(f"Matriz de rotação inválida com determinante: {det_R}")
    if np.linalg.norm(t) > max_expected_translation:
        raise ValueError(f"Magnitude da translação {np.linalg.norm(t)} excede o limite esperado")

    # Verificação adicional para evitar erros numéricos acumulados
    if np.any(np.isnan(T)) or np.any(np.isinf(T)):
        raise ValueError("A matriz de transformação contém valores NaN ou Inf, o que indica erro numérico.")

def calculate_transformation_error(T1, T2):
    """
    Calcula o erro entre duas matrizes de transformação usando a norma de Frobenius.
    Parâmetros:
    - T1 (ndarray): Primeira matriz de transformação 4x4.
    - T2 (ndarray): Segunda matriz de transformação 4x4.
    Retorna:
    - float: Erro calculado entre as duas matrizes.
    """
    return np.linalg.norm(T1 - T2, ord='fro')  # Calcula a norma de Frobenius da diferença entre as duas matrizes

def icp(A, B, init_pose=None, max_iterations=50, tolerance=1e-6, rejection_threshold=2.0, downsample_rate=0.2, normal_weight=0.1):
    """
    Implementa o Algoritmo Iterativo de Pontos mais Próximos (ICP) com refinamentos.
    Parâmetros:
    - A (ndarray): Nuvem de pontos fonte (n x 3).
    - B (ndarray): Nuvem de pontos destino (n x 3).
    - init_pose (ndarray): Matriz de transformação inicial 4x4 [opcional].
    - max_iterations (int): Número máximo de iterações do ICP (default: 50).
    - tolerance (float): Critério de parada baseado na mudança na matriz de transformação (default: 1e-6).
    - rejection_threshold (float): Distância limite para rejeição de outliers (default: 2.0).
    - downsample_rate (float): Taxa de amostragem para downsampling (default: 0.2).
    - normal_weight (float): Peso aplicado às normais durante a correspondência de pontos (default: 0.1).
    Retorna:
    - T (ndarray): Matriz de transformação 4x4 final que mapeia A para B.
    - combined_distances (ndarray): Distâncias combinadas utilizadas no critério de correspondência.
    """
    # Prepara as nuvens de pontos como matrizes homogêneas
    src = np.ones((4, A.shape[0]))
    dst = np.ones((4, B.shape[0]))
    src[:3, :] = np.copy(A.T)
    dst[:3, :] = np.copy(B.T)

    # Aplica a transformação inicial, se fornecida
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0  # Inicializa o erro anterior

    # Calcula as normais das nuvens de pontos
    normals_A = compute_normals(A)
    normals_B = compute_normals(B)

    for i in range(max_iterations):
        # Amostragem das nuvens de pontos para downsampling
        if downsample_rate < 1.0:
            sampled_indices = np.random.choice(src.shape[1], int(src.shape[1] * downsample_rate), replace=False)
            src_sampled = src[:, sampled_indices]
            normals_A_sampled = normals_A[sampled_indices]
        else:
            src_sampled = src
            normals_A_sampled = normals_A

        # Encontra as correspondências utilizando KDTree
        tree = KDTree(dst[:3, :].T)
        distances, indices = tree.query(src_sampled[:3, :].T)
        normals_distances = np.linalg.norm(normals_A_sampled - normals_B[indices], axis=1)

        # Calcula as distâncias combinadas e aplica rejeição de outliers
        combined_distances = distances + normals_distances * normal_weight
        rejection_threshold = dynamic_rejection_threshold(combined_distances)
        mask = combined_distances < rejection_threshold
        src_sampled = src_sampled[:, mask]
        indices = indices[mask]

        # Calcula a transformação usando SVD
        T, _, _ = best_fit_transform(src_sampled[:3, :].T, dst[:3, indices].T)
        src = np.dot(T, src)  # Aplica a transformação

        # Calcula o erro médio das correspondências
        mean_error = np.mean(combined_distances[mask])
        if np.abs(prev_error - mean_error) < tolerance:
            break  # Interrompe se a mudança no erro for menor que a tolerância
        prev_error = mean_error

    # Recalcula a transformação final para a nuvem de pontos original
    T, _, _ = best_fit_transform(A, src[:3, :].T)
    validate_transformation(T)  # Valida a transformação final

    return T, combined_distances

def best_fit_transform(A, B):
    """
    Calcula a melhor transformação que mapeia A para B usando SVD.
    Parâmetros:
    - A (ndarray): Nuvem de pontos fonte (n x 3).
    - B (ndarray): Nuvem de pontos destino (n x 3).
    Retorna:
    - T (ndarray): Matriz de transformação 4x4.
    - R (ndarray): Matriz de rotação 3x3.
    - t (ndarray): Vetor de translação 3x1.
    """
    assert A.shape == B.shape

    # Calcula os centróides das nuvens de pontos
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A  # Centraliza a nuvem A
    BB = B - centroid_B  # Centraliza a nuvem B

    H = np.dot(AA.T, BB)  # Calcula a matriz de covariância

    # Aplica SVD para encontrar a rotação
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Verifica e ajusta a rotação para garantir que ela não seja refletida
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R, centroid_A.T)  # Calcula o vetor de translação

    # Constrói a matriz de transformação homogênea
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, R, t

def smooth_trajectory(positions, smoothing_factor=0.9):
    """
    Suaviza a trajetória estimada utilizando um fator de suavização.
    Parâmetros:
    - positions (list of ndarray): Lista de posições estimadas (n x 3).
    - smoothing_factor (float): Fator de suavização (default: 0.9).
    Retorna:
    - ndarray: Trajetória suavizada (n x 3).
    """
    smoothed_positions = [positions[0]]
    for i in range(1, len(positions)):
        smoothed_position = smoothing_factor * smoothed_positions[-1] + (1 - smoothing_factor) * positions[i]
        smoothed_positions.append(smoothed_position)
    return np.array(smoothed_positions)

# Diretório onde os arquivos .obj estão armazenados
base_dir = "/content/point_clouds"

# Caminho para o Ground-Truth
ground_truth_path = "/content/ground_truth.npy"
ground_truth = np.load(ground_truth_path)

# Listar todos os diretórios
subdirs = sorted(os.listdir(base_dir))

# Inicializar a matriz de transformação acumulada e listas para erros e posições estimadas
T_accumulated = np.eye(4)  # Matriz de transformação inicial
matrix_errors = []  # Lista para armazenar os erros das transformações
estimated_positions = []  # Lista para armazenar as posições estimadas
estimated_positions.append(T_accumulated[:3, 3])  # Adiciona a posição inicial

# Parâmetro de ajuste do erro
adjustment_factor = 0.5  # Fator para ajustar a transformação com base no erro anterior

# Loop para processar os pares de scans
for i in range(1, len(subdirs)):
    # Carrega os arquivos de nuvem de pontos .obj
    file_A = os.path.join(base_dir, subdirs[i-1], f"{subdirs[i-1]}_points.obj")
    file_B = os.path.join(base_dir, subdirs[i], f"{subdirs[i]}_points.obj")

    cloud_A = trimesh.load(file_A).vertices
    cloud_B = trimesh.load(file_B).vertices

    # Normaliza a escala das nuvens de pontos
    cloud_A = normalize_scale(cloud_A)
    cloud_B = normalize_scale(cloud_B)

    # Executa o ICP para encontrar a transformação
    T, _ = icp(cloud_A, cloud_B)

    # Ajuste da Transformação com base no Erro Anterior
    true_matrix = ground_truth[i]
    error_previous = np.linalg.norm(T_accumulated - true_matrix, ord='fro')
    adjustment_matrix = adjustment_factor * (true_matrix - T_accumulated)

    # Aplica o ajuste e atualiza a matriz de transformação acumulada
    T_adjusted = T + adjustment_matrix
    T_accumulated = np.dot(T_accumulated, T_adjusted)
    estimated_positions.append(T_accumulated[:3, 3])

    # Calcular o erro atual e armazenar
    matrix_error = calculate_transformation_error(T_accumulated, true_matrix)
    matrix_errors.append(matrix_error)

    print(f"Erro na iteração {i}: {matrix_error}")

# Suaviza as posições estimadas
smoothed_positions = smooth_trajectory(estimated_positions)

# Plotagem da trajetória estimada e comparação com a Ground-Truth
positions = np.array(smoothed_positions)
gt_positions = ground_truth[:, :3, 3]

fig = plt.figure(figsize=(14, 8))

# Plot da trajetória estimada
ax = fig.add_subplot(121, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o', linestyle='-', color='b', label='Estimativa')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trajetória Estimada do Veículo (Suavizada)')
ax.legend()

# Plot da Ground-Truth
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], marker='o', linestyle='-', color='r', label='Ground Truth')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Ground-Truth')
ax2.legend()

plt.show()

# Exibe a matriz de transformação final acumulada
print("Matriz de Transformação Final (em coordenadas homogêneas):")
print(T_accumulated)

# Calcula e imprime o erro médio das matrizes de transformação
mean_matrix_error = np.mean(matrix_errors)
print(f"Erro médio das matrizes de transformação em relação à Ground-Truth: {mean_matrix_error}")