import numpy as np


class DensityEvolution:

    def __init__(self, epsilon, protograph, iterations=50):
        matrix = np.asarray(protograph)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError("Протограф должен быть непустой двумерной матрицей")
        if matrix.dtype.kind not in "biuf":
            raise ValueError("Элементы матрицы должны быть вещественными числами из [0, 1]")
        if not np.all(np.isfinite(matrix)) or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError("Элементы матрицы должны быть конечными числами из [0, 1]")
        if not np.any(matrix > 0):
            raise ValueError("Матрица должна содержать хотя бы одну ненулевую связь")

        channel = np.asarray(epsilon)
        if channel.ndim != 0 or channel.dtype.kind not in "biuf":
            raise ValueError("epsilon должна быть вещественным числом из [0, 1]")
        if not np.isfinite(channel) or not 0 <= channel <= 1:
            raise ValueError("epsilon должна находиться в диапазоне [0, 1]")
        if (
            isinstance(iterations, (bool, np.bool_))
            or not isinstance(iterations, (int, np.integer))
            or iterations < 0
        ):
            raise ValueError("iterations должно быть неотрицательным целым числом")

        self.epsilon = float(channel)
        self.protograph = matrix.astype(float)
        self.iterations = int(iterations)

    def de(self, *, return_logs=False, return_edge_messages=False):

        matrix = self.protograph
        edges = matrix > 0
        with np.errstate(divide="ignore"):
            log_epsilon = np.log(self.epsilon)
            log_h = np.log(matrix)
            log_one_minus_h = np.log1p(-matrix)

        log_x = np.full(edges.shape, -np.inf)
        log_x[edges] = log_epsilon
        log_p = log_x.copy()
        log_messages = None
        for _ in range(self.iterations):
            # Шаг 1. CN -> VN: исключаем целевой символ j.
            log_messages = np.full(edges.shape, -np.inf)
            for i, j in zip(*np.nonzero(edges)):
                neighbors = edges[i].copy()
                neighbors[j] = False
                incoming = log_h[i, neighbors] + log_x[i, neighbors]
                if incoming.size == 0:
                    continue  # Проверка степени 1 восстанавливает бит.

                complements = self._log1mexp(incoming)
                log_product = complements.sum()
                if log_product <= -np.log(2.0):
                    log_messages[i, j] = self._log1mexp(log_product)
                else:
                    # Устойчивый расчёт 1 - произведение(1 - h_is * P_is).
                    prefix = np.concatenate(([0.0], np.cumsum(complements[:-1])))
                    log_messages[i, j] = min(
                        0.0, np.logaddexp.reduce(incoming + prefix)
                    )

            # Шаг 2. Условное сообщение проверки: h*y + (1-h).
            with np.errstate(under="ignore"):
                log_factors = np.logaddexp(log_h + log_messages, log_one_minus_h)
            log_factors = np.minimum(log_factors, 0.0)

            # Апостериорная оценка символа: все соседние проверки.
            log_p_col = log_epsilon + log_factors.sum(axis=0)
            log_p = np.full(edges.shape, -np.inf)
            log_p[edges] = np.broadcast_to(log_p_col, edges.shape)[edges]

            # Шаг 3. VN -> CN: исключаем целевую проверку i.
            # Prefix/suffix корректны и при factor=0 (log_factor=-inf):
            # при вычитании из общей суммы возникло бы -inf-(-inf)=nan.
            prefix = np.zeros_like(log_factors)
            suffix = np.zeros_like(log_factors)
            prefix[1:, :] = np.cumsum(log_factors[:-1, :], axis=0)
            suffix[:-1, :] = np.cumsum(log_factors[:0:-1, :], axis=0)[::-1, :]
            log_x_next = log_epsilon + prefix + suffix
            log_x = np.full(edges.shape, -np.inf)
            log_x[edges] = log_x_next[edges]

        if return_logs:
            result = (log_p, log_messages)
            return (*result, log_x) if return_edge_messages else result

        with np.errstate(under="ignore"):
            p = np.exp(log_p)
            messages = None if log_messages is None else np.exp(log_messages)
            if return_edge_messages:
                return p, messages, np.exp(log_x)
        return p, messages

    @staticmethod
    def _log1mexp(x):
        
        x = np.asarray(x, dtype=float)
        result = np.empty_like(x)
        small = x < -np.log(2.0)
        with np.errstate(divide="ignore", under="ignore"):
            result[small] = np.log1p(-np.exp(x[small]))
            result[~small] = np.log(-np.expm1(x[~small]))
        return result
