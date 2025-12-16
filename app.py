from flask import Flask, render_template, request
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import io
import base64
from pathlib import Path

PDF_PATH = "/mnt/data/System_all_r_is_0.pdf"

app = Flask(__name__)
app.config['SECRET_KEY'] = '89033838145'


class Attr:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def val(self):
        return float(self.value)


class F:
    def __init__(self, a, b, c, d, L):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.L = L

    def calc(self, x):
        return (self.a * (x ** 3)) + (self.b * (x ** 2)) + (self.c * x) + (self.d)


# ---------------------------
# Наименования переменных
# ---------------------------
v0 = {
    'B₁': Attr('Степень разработанности технического задания на проведение проектных работ', 0.1),
    'B₂': Attr('Качество проведения проектных работ', 0.2),
    'B₃': Attr('Эффективность мониторинга и контроля за проведением проектных работ', 0.8),
    'B₄': Attr('Уровень квалификации команды', 0.15),
    'B₅': Attr('Эффективность распределения ограниченных ресурсов Ресурсоэффективность', 0.3),
    'B₆': Attr('Адекватность оценки рисков Оценка рисков', 0.4),
    'B₇': Attr('Безопасность ПО', 0.6),
    'B₈': Attr('Целостность ПО', 0.25),
    'B₉': Attr('Системность ПО', 0.05),
    'B₁₀': Attr('Прозрачность коммуникации', 0.7),
    'B₁₁': Attr('Функциональность ПО', 0.35),
    'B₁₂': Attr('Надежность ПО', 0.45),
    'B₁₃': Attr('Производительность ПО', 0.5),
    'B₁₄': Attr('Сопровождаемость ПО', 0.2),
    'B₁₅': Attr('Переносимость По', 0.3),
    'B₁₆': Attr('Тестопригодность по', 0.25),
    'B₁₇': Attr('Понятность по', 0.35)
}

c = {
    'B₁*': Attr('Коэффициент Степень разработанности технического задания на проведение проектных работ', 1.0),
    'B₂*': Attr('Коэффициент Качество проведения проектных работ', 1.0),
    'B₃*': Attr('Коэффициент Эффективность мониторинга и контроля за проведением проектных работ', 1.0),
    'B₄*': Attr('Коэффициент Уровень квалификации команды', 1.0),
    'B₅*': Attr('Коэффициент Эффективность распределения ограниченных ресурсов Ресурсоэффективность', 1.0),
    'B₆*': Attr('Коэффициент Адекватность оценки рисков Оценка рисков', 1.0),
    'B₇*': Attr('Коэффициент Безопасность ПО', 1.0),
    'B₈*': Attr('Коэффициент Целостность ПО', 1.0),
    'B₉*': Attr('Коэффициент Системность ПО', 1.0),
    'B₁₀*': Attr('Коэффициент Прозрачность коммуникации', 1.0),
    'B₁₁*': Attr('Коэффициент Функциональность ПО', 1.0),
    'B₁₂*': Attr('Коэффициент Надежность ПО', 1.0),
    'B₁₃*': Attr('Коэффициент Производительность ПО', 1.0),
    'B₁₄*': Attr('Коэффициент Сопровождаемость ПО', 1.0),
    'B₁₅*': Attr('Коэффициент Переносимость По', 1.0),
    'B₁₆*': Attr('Коэффициент Тестопригодность по', 1.0),
    'B₁₇*': Attr('Коэффициент Понятность по', 1.0)
}


# ---------------------------
# Инициализация f1..f128 с улучшенными коэффициентами
# ---------------------------
def subscript_num(n):
    sub = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    return ''.join(sub[ch] for ch in str(n))


# Сбалансированные коэффициенты как в эпидемиологической модели
f = {}
for i in range(1, 129):
    param_num = (i % 17) + 1
    L = f"B{subscript_num(param_num)}"
    # Сбалансированные коэффициенты для предотвращения выхода за границы [0,1]
    if i % 5 == 0:
        f[f'f{i}'] = F(0.05, -0.08, 0.3, 0.1, L)
    elif i % 5 == 1:
        f[f'f{i}'] = F(0.08, -0.12, 0.35, 0.05, L)
    elif i % 5 == 2:
        f[f'f{i}'] = F(0.04, -0.06, 0.25, 0.08, L)
    elif i % 5 == 3:
        f[f'f{i}'] = F(0.1, -0.15, 0.4, 0.05, L)
    else:
        f[f'f{i}'] = F(0.06, -0.09, 0.3, 0.1, L)

# Описания функций как в эпидемиологической модели
f_descriptions = {}
for i in range(1, 129):
    param_num = (i % 17) + 1
    param_name = v0[f'B{subscript_num(param_num)}'].name
    f_descriptions[f'f{i}'] = f"Влияние на {param_name}"


# ---------------------------
# Функции управления a1..a5 с ограничениями
# ---------------------------
def bounded_value(value):
    return max(0.0, min(1.0, value))


def a1(t):
    return bounded_value(0.3 + 0.1 * t + 0.05 * np.sin(3 * t))


def a2(t):
    return bounded_value(0.4 + 0.15 * t + 0.03 * np.cos(4 * t))


def a3(t):
    return bounded_value(0.2 + 0.08 * t + 0.07 * np.sin(2 * t))


def a4(t):
    return bounded_value(0.5 + 0.2 * t + 0.04 * np.cos(5 * t))


def a5(t):
    return bounded_value(0.35 + 0.12 * t + 0.06 * np.sin(6 * t))


# ---------------------------
# Соотношение gN -> соответствующая aK (по PDF)
# ---------------------------
g_map = {
    1: 5, 2: 2, 3: 2, 4: 3, 5: 4, 6: 5, 7: 2, 8: 5, 9: 3, 10: 5,
    11: 3, 12: 5, 13: 1, 14: 4, 15: 1, 16: 3, 17: 4, 18: 2, 19: 5, 20: 2,
    21: 1, 22: 5, 23: 2, 24: 5, 25: 3, 26: 5, 27: 3, 28: 4, 29: 3, 31: 1,
    32: 2, 33: 5, 34: 2, 35: 5, 36: 4, 37: 1, 38: 3, 39: 5, 40: 4, 41: 2,
    42: 1, 43: 3
}


def g_eval(gnum, t):
    k = g_map.get(gnum, None)
    if k == 1:
        return a1(t)
    elif k == 2:
        return a2(t)
    elif k == 3:
        return a3(t)
    elif k == 4:
        return a4(t)
    elif k == 5:
        return a5(t)
    else:
        return 1.0


# ---------------------------
# Структура уравнений
# ---------------------------
eq_structure = {
    1: {'left_f': [(1, 2), (2, 3), (4, 17)], 'left_g': [1], 'right_f': [(3, 15), (5, 7), (6, 13), (7, 14), (8, 16)],
        'right_g': [2]},
    2: {'left_f': [(9, 3), (10, 5), (12, 10), (13, 11)], 'left_g': [3, 4], 'right_f': [(11, 7), (14, 4)],
        'right_g': [5, 6]},
    3: {'left_f': [(15, 4), (16, 14), (17, 17)], 'left_g': [7], 'right_f': [(18, 7)], 'right_g': [8]},
    4: {'left_f': [(19, 3), (20, 7), (21, 16)], 'left_g': [9],
        'right_f': [(22, 5), (23, 6), (24, 8), (25, 9), (26, 15)], 'right_g': [10]},
    5: {'left_f': [(27, 6), (28, 8), (29, 10), (30, 12), (31, 13), (32, 15), (33, 16), (34, 17)], 'left_g': [11, 12],
        'right_f': [(35, 2), (36, 4)], 'right_g': [13]},
    6: {'left_f': [(38, 5), (39, 9), (40, 11), (41, 15)], 'left_g': [14], 'right_f': [(37, 4), (42, 1)],
        'right_g': [15]},
    7: {'left_f': [(43, 2)], 'left_g': [15, 16, 17],
        'right_f': [(44, 3), (45, 4), (46, 5), (47, 8), (48, 9), (49, 11), (50, 14), (51, 15)], 'right_g': [18]},
    8: {'left_f': [(52, 2), (53, 10), (54, 12), (55, 13), (56, 15), (57, 16)], 'left_g': [19], 'right_f': [(58, 7)],
        'right_g': [20]},
    9: {'left_f': [(59, 6), (60, 8), (61, 10), (62, 11), (63, 13), (64, 14), (65, 15), (66, 16)], 'left_g': [21, 22],
        'right_f': [(67, 4), (68, 7)], 'right_g': [23]},
    10: {'left_f': [(69, 2), (70, 8), (71, 9), (72, 11), (73, 12), (74, 13), (75, 15), (76, 16)], 'left_g': [24],
         'right_f': [], 'right_g': [25]},
    11: {'left_f': [(77, 1), (78, 6), (79, 10), (80, 15)], 'left_g': [25],
         'right_f': [(81, 2), (82, 5), (83, 7), (84, 8), (85, 14)], 'right_g': [27]},
    12: {'left_f': [(86, 1), (87, 3), (88, 5), (89, 8), (90, 10), (91, 13), (92, 14), (93, 16)], 'left_g': [28],
         'right_f': [(94, 17)], 'right_g': [29]},
    13: {'left_f': [(95, 6), (96, 8), (97, 10), (98, 11), (99, 12), (100, 15), (101, 16)], 'left_g': [31, 32],
         'right_f': [(102, 4), (103, 9), (104, 17)], 'right_g': [33]},
    14: {'left_f': [(105, 15), (106, 17)], 'left_g': [34, 35], 'right_f': [(107, 6)], 'right_g': [36]},
    15: {'left_f': [(108, 1), (109, 5), (110, 6), (111, 8), (112, 9), (113, 10), (114, 11), (115, 13), (116, 16),
                    (117, 17)], 'left_g': [37, 38, 39], 'right_f': [(118, 2), (119, 4), (120, 7)], 'right_g': [40]},
    16: {'left_f': [(121, 3), (122, 14)], 'left_g': [41], 'right_f': [(123, 5), (124, 15)], 'right_g': [42]},
    17: {'left_f': [(125, 1), (126, 12)], 'left_g': [43], 'right_f': [(127, 2), (128, 11)], 'right_g': []}
}


def prod(iterable):
    result = 1.0
    for x in iterable:
        result *= x
    return result


def smooth_boundary(u_val, derivative):
    """Плавное ограничение производных у границ как в эпидемиологической модели"""
    if u_val >= 0.9 and derivative > 0:
        return derivative * (1.0 - u_val) * 10
    elif u_val <= 0.1 and derivative < 0:
        return derivative * u_val * 10
    elif u_val >= 0.7 and derivative > 0:
        return derivative * 0.5
    elif u_val <= 0.3 and derivative < 0:
        return derivative * 0.5
    return derivative


def new_du_dt(u, t):
    B = [max(0.0, min(1.0, x)) for x in u]  # Гарантируем границы [0,1]
    derivatives = []

    for i in range(1, 18):
        struct = eq_structure[i]

        # Левая часть: f(...)*g(...)
        left_f_vals = []
        for fnum, Bnum in struct['left_f']:
            key = f'f{fnum}'
            left_f_vals.append(f[key].calc(B[Bnum - 1]))
        left_g_vals = [g_eval(gnum, t) for gnum in struct['left_g']]
        left_prod = prod(left_f_vals) * prod(left_g_vals)

        # Правая часть
        right_f_vals = []
        for fnum, Bnum in struct['right_f']:
            key = f'f{fnum}'
            right_f_vals.append(f[key].calc(B[Bnum - 1]))
        right_g_vals = [g_eval(gnum, t) for gnum in struct['right_g']]
        right_prod = prod(right_f_vals) * prod(right_g_vals)

        denom_key = f'B{subscript_num(i)}*'
        denom = max(0.5, c.get(denom_key, Attr(denom_key, 1.0)).val())

        db_dt = (1.0 / denom) * (left_prod - right_prod)

        # Применяем плавное ограничение
        bounded_deriv = smooth_boundary(B[i - 1], db_dt)
        bounded_deriv = max(-0.5, min(0.5, bounded_deriv))
        derivatives.append(bounded_deriv)

    return derivatives


@app.route('/', methods=['GET', 'POST'])
def index():
    t_span = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    plot_data_list = []
    polar_plot_data = None
    show_polar = False

    if request.method == 'POST':
        if any(k.startswith('norm_bound_') for k in request.form.keys()):
            show_polar = True
        else:
            show_polar = False

        # Обновляем начальные значения
        for key in v0:
            try:
                v0[key].value = float(request.form.get(f'v0_{key}', v0[key].value))
            except:
                pass

        # Обновляем коэффициенты нормализации
        for key in c:
            try:
                c[key].value = float(request.form.get(f'c_{key}', c[key].value))
            except:
                pass

        # Обновляем параметры f
        for i in range(1, 129):
            fk = f'f{i}'
            if fk in f:
                try:
                    f[fk].a = float(request.form.get(f'{fk}_a', f[fk].a))
                    f[fk].b = float(request.form.get(f'{fk}_b', f[fk].b))
                    f[fk].c = float(request.form.get(f'{fk}_c', f[fk].c))
                    f[fk].d = float(request.form.get(f'{fk}_d', f[fk].d))
                except:
                    pass

        try:
            t0 = [max(0.0, min(1.0, v0[key].value)) for key in
                  ['B₁', 'B₂', 'B₃', 'B₄', 'B₅', 'B₆', 'B₇', 'B₈', 'B₉', 'B₁₀', 'B₁₁', 'B₁₂', 'B₁₃', 'B₁₄', 'B₁₅',
                   'B₁₆', 'B₁₇']]

            # Основной график
            if not show_polar:
                t_span_main = np.arange(0.0, 2.0, 0.02)  # Увеличили время для лучшей динамики
                sol = odeint(new_du_dt, t0, t_span_main)
                sol = np.maximum(sol, 0.0)
                sol = np.minimum(sol, 1.0)

                # Подробные подписи для легенды как в эпидемиологической модели
                labels_short = ['B₁', 'B₂', 'B₃', 'B₄', 'B₅', 'B₆', 'B₇', 'B₈', 'B₉', 'B₁₀',
                                'B₁₁', 'B₁₂', 'B₁₃', 'B₁₄', 'B₁₅', 'B₁₆', 'B₁₇']
                labels_detailed = [
                    'B₁ - Степень разработанности технического задания',
                    'B₂ - Качество проведения проектных работ',
                    'B₃ - Эффективность мониторинга и контроля',
                    'B₄ - Уровень квалификации команды',
                    'B₅ - Эффективность распределения ресурсов',
                    'B₆ - Адекватность оценки рисков',
                    'B₇ - Безопасность ПО',
                    'B₈ - Целостность ПО',
                    'B₉ - Системность ПО',
                    'B₁₀ - Прозрачность коммуникации',
                    'B₁₁ - Функциональность ПО',
                    'B₁₂ - Надежность ПО',
                    'B₁₃ - Производительность ПО',
                    'B₁₄ - Сопровождаемость ПО',
                    'B₁₅ - Переносимость ПО',
                    'B₁₆ - Тестопригодность ПО',
                    'B₁₇ - Понятность ПО'
                ]

                colors = plt.cm.tab20(np.linspace(0, 1, 17))

                # Разделяем на 3 группы как в эпидемиологической модели
                groups = [
                    (0, 6, "График 1: Параметры B₁-B₆"),
                    (6, 12, "График 2: Параметры B₇-B₁₂"),
                    (12, 17, "График 3: Параметры B₁₃-B₁₇")
                ]

                for start_idx, end_idx, title in groups:
                    fig, ax = plt.subplots(figsize=(16, 8))

                    for i in range(start_idx, end_idx):
                        y = sol[:, i]
                        ax.plot(t_span_main, y, color=colors[i], linewidth=2, label=labels_detailed[i])
                        ax.text(
                            t_span_main[-1] + 0.01,
                            y[-1],
                            labels_short[i],
                            fontsize=10,
                            color='black',
                            va='center'
                        )

                    ax.set_xlabel('Время', fontsize=12)
                    ax.set_ylabel('Значения', fontsize=12)
                    ax.set_title(title, fontsize=14)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.0)
                    ax.set_xlim([0, 2.05])
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    plot_data_list.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
                    plt.close(fig)

            # Полярные диаграммы
            if show_polar:
                norm_bounds = []
                for i in range(17):
                    try:
                        nb = float(request.form.get(f'norm_bound_{i}', 1.0))
                    except:
                        nb = 1.0
                    norm_bounds.append(max(0.0, min(1.0, nb)))

                t_span_polar = np.array([0, 0.4, 0.8, 1.2, 1.6, 2.0])
                sol_polar = odeint(new_du_dt, t0, t_span_polar)
                sol_polar = np.maximum(sol_polar, 0.0)
                sol_polar = np.minimum(sol_polar, 1.0)

                fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': 'polar'})
                axes = axes.flatten()

                labels = ['B₁', 'B₂', 'B₃', 'B₄', 'B₅', 'B₆', 'B₇', 'B₈', 'B₉', 'B₁₀',
                          'B₁₁', 'B₁₂', 'B₁₃', 'B₁₄', 'B₁₅', 'B₁₆', 'B₁₇']
                angles = np.linspace(0, 2 * np.pi, 17, endpoint=False)
                angles_closed = np.append(angles, angles[0])
                norm_bounds_closed = np.append(norm_bounds, norm_bounds[0])

                for i, ax in enumerate(axes):
                    if i < len(t_span_polar):
                        sol_values = np.append(sol_polar[i, :], sol_polar[i, 0])
                        ax.plot(angles_closed, sol_values, linewidth=2)
                        ax.fill(angles_closed, sol_values, alpha=0.25)
                        ax.plot(angles_closed, norm_bounds_closed, linestyle='--', linewidth=1)
                        ax.fill(angles_closed, norm_bounds_closed, alpha=0.1)
                        ax.set_xticks(angles)
                        ax.set_xticklabels(labels)
                        ax.set_ylim(0, 1.0)
                        ax.set_title(f'Время t = {t_span_polar[i]}', fontsize=12, pad=20)
                    else:
                        ax.axis('off')

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                polar_plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close(fig)

        except Exception as e:
            print(f"Error generating plot: {e}")

    # Группируем функции по 2 на строку для отображения как в эпидемиологической модели
    f_items = list(f.items())[:20]  # Только первые 20 функций
    f_grouped = []
    for i in range(0, len(f_items), 2):
        if i + 1 < len(f_items):
            f_grouped.append([f_items[i], f_items[i + 1]])
        else:
            f_grouped.append([f_items[i]])

    return render_template('index.html', v0=v0, c=c, f=f, f_grouped=f_grouped, t_span=t_span,
                           plot_data_list=plot_data_list, polar_plot_data=polar_plot_data,
                           show_polar=show_polar, f_descriptions=f_descriptions)


if __name__ == '__main__':
    app.run(debug=True)