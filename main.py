import numpy as np
import cv2
import mediapipe as mp
from math import hypot
import time


def draw_finger(cord):
    """draws finger on the screen and returns its coordinates"""
    x_tip = int(results.multi_hand_landmarks[0].landmark[cord].x *
                printout.shape[1])
    y_tip = int(results.multi_hand_landmarks[0].landmark[cord].y *
                printout.shape[0])
    cv2.circle(printout, (x_tip, y_tip), 10, (255, 0, 0), -1)
    return (x_tip, y_tip)


def solve_exp(s):
    """solves given string as an expression"""
    try:
        temp = eval(s)
        return str(temp)
    except SyntaxError:
        return s
    except ZeroDivisionError:
        return 1


def get_rdist(s, f, res: 'results.multi_hand_landmarks[0].landmark'):
    """returns relative distance between the fingers"""
    cord1 = (res[17].x - res[5].x) * flippedRGB.shape[1]
    cord2 = (res[17].y - res[5].y) * flippedRGB.shape[0]
    finger_dist = hypot(cord1, cord2)
    hand_dist = hypot(s[0] - f[0], s[1] - f[1])
    if hand_dist == 0:
        return 10e9
    return finger_dist / hand_dist


def draw_line(s, f, color):
    """draws the line"""
    cv2.line(printout, s, f, color, thickness=5)


def draw_calculator(mat):
    """draws the calculator"""

    def draw_text():
        cv2.putText(mat, "1", (10, 230), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "2", (10 + 80, 230), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "3", (10 + 160, 230), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "4", (10, 230 + 80), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "5", (10 + 80, 230 + 80), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "6", (10 + 160, 230 + 80), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "7", (10, 230 + 160), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "8", (10 + 80, 230 + 160), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "9", (10 + 160, 230 + 160), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "0", (10 + 0, 230 + 240), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "/", (15 + 80, 225 + 240), 1, 5, (255, 0, 0), thickness=6)
        cv2.putText(mat, ".", (25 + 160, 215 + 240), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "=", (5 + 240, 230 + 240), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "+", (5 + 240, 230 + 160), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "-", (5 + 240, 230 + 80), 1, 6, (255, 0, 0), thickness=6)
        cv2.putText(mat, "*", (10 + 240, 230), 1, 6, (255, 0, 0), thickness=6)

    mat[:480, :320] = np.ones((480, 320, 3)) * 255
    for i in range(1, 5):
        cv2.line(mat, (0, STEP * (i + 1)), (320, STEP * (i + 1)), (0, 0, 0), 2)
        cv2.line(mat, (STEP * i, STEP * 2), (STEP * i, 480), (0, 0, 0), 2)
    draw_text()

def draw_exp(e):
    def find_thick(n):
        size = 5
        if 7 <= n < 8:
            size -= 1
        elif 8 <= n < 11:
            size -= 2
        elif 11 <= n < 16:
            size -= 3
        elif 16 <= n:
            size -= 4
        return size
    cv2.putText(printout, e, (10, 140), 1, find_thick(len(e)), (255, 0, 0), thickness=2)


def get_touch(c1, c2):
    global exp
    c = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
    val = -1
    ns = False
    for i in range(16):
        if 0 < c[0] - BUTTONS[i][0] < 80 and 0 < c[1] - BUTTONS[i][1] < 80:
            val = i
    sign = BUTTONS[val][2]
    if exp != '' and solve_exp(exp) == 1:
        ns = True
    if exp == '' and sign not in '+-*/.=':
        exp += sign
    elif (exp.rfind('.') > max(exp.rfind('+'), exp.rfind('-'), exp.rfind('*'), exp.rfind('/')) and sign == '.' and exp.rfind('.') != -1) or (sign in '-+/*=' and ns) or val == -1 or (len(exp) >= 2 and (exp[-2:] == '/0' and sign in '+-*/=')) or (exp == '' and sign in '+-*/.=') or (sign in '+-*/' and exp[-1] == '.') or (
            sign == '.' and exp[-1] in '+-*/') or (sign == '=' and exp[-1] == '.'):
        return 0
    elif exp == '':
        exp += sign
    elif sign == '=':
        exp = solve_exp(exp)
    elif sign in '+-*/' and exp[-1] in '+-*/':
        exp = exp[:-1] + sign
    else:
        exp += sign


TOUCH = 2.5
LAG = 1
STEP = 80
BUTTONS = [(0, 400, '0'), (0, 160, '1'), (80, 160, '2'), (160, 160, '3'), (0, 240, '4'),
           (80, 240, '5'), (160, 240, '6'), (0, 320, '7'), (80, 320, '8'), (160, 320, '9'),
           (240, 160, '*'), (240, 240, '-'), (240, 320, '+'), (240, 400, '='), (160, 400, '.'), (80, 400, '/')]
exp = ''
curr = time.time()
line_color = (0, 0, 255)

handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    printout = flippedRGB.copy()
    res_image = cv2.cvtColor(printout, cv2.COLOR_RGB2BGR)
    draw_calculator(printout)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        cordfirst = draw_finger(8)
        cordsec = draw_finger(12)
        dist = get_rdist(cordfirst, cordsec, results.multi_hand_landmarks[0].landmark)
        if dist >= TOUCH:
            line_color = (0, 255, 0)
            if time.time() - curr > LAG:
                curr = time.time()
                get_touch(cordfirst, cordsec)
        draw_line(cordfirst, cordsec, color=line_color)
        line_color = (0, 0, 255)
    draw_exp(exp)
    res_image = cv2.cvtColor(printout, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)