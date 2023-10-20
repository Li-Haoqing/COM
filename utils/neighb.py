import numpy as np


def neighb3(cx, cy, e):
    new_e = np.array([])
    for p in e:
        if (cy * 4) < p < cy * (cy - 4):
            if p != (cy * cy - 1) and (p + 1) not in e:
                if (p - cy) in e:
                    if (p + cy) not in e:
                        p1 = (p - 3, p - 2, p - 1, p,
                              p - cx - 3, p - cx - 2, p - cx - 1, p - cx,
                              p - 2 * cx - 3, p - 2 * cx - 2, p - 2 * cx - 1, p - 2 * cx,
                              p - 3 * cx - 3, p - 3 * cx - 2, p - 3 * cx - 1, p - 3 * cx)
                        new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), p1)))
                    else:
                        new_e = np.append(new_e,
                                          list(filter(lambda point: point < (cx * cy), (p - 3, p - 2, p - 1, p))))
                else:
                    new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p - 3, p - 2, p - 1, p))))

            if p != 0 and (p - 1) not in e:
                if (p - cy) in e:
                    if (p + cy) not in e:
                        p2 = (p, p + 1, p + 2, p + 3,
                              p + cx, p + cx + 1, p + cx + 2, p + cx + 3,
                              p + 2 * cx, p + 2 * cx + 1, p + 2 * cx + 2, p + 2 * cx + 3,
                              p + 3 * cx, p + 3 * cx + 1, p + 3 * cx + 2, p + 3 * cx + 3)
                        new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), p2)))
                    else:
                        new_e = np.append(new_e,
                                          list(filter(lambda point: point < (cx * cy), (p, p + 1, p + 2, p + 3))))
                else:
                    new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p, p + 1, p + 2, p + 3))))

            if (p - cy) not in e:
                new_e = np.append(new_e,
                                  list(filter(lambda point: point < (cx * cy), (p, p + cx, p + 2 * cx, p + 3 * cx))))
    return new_e.astype('int64')


def neighb2(cx, cy, e):
    new_e = np.array([])
    for p in e:
        if (cy * 4) < p < cy * (cy - 4):
            if p != (cy * cy - 1) and (p + 1) not in e:
                if (p - cy) in e:
                    if (p + cy) not in e:
                        p1 = (p - 2, p - 1, p,
                              p - cx - 2, p - cx - 1, p - cx,
                              p - 2 * cx - 2, p - 2 * cx - 1, p - 2 * cx,
                              p - 3 * cx - 2, p - 3 * cx - 1, p - 3 * cx)
                        new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), p1)))
                    else:
                        new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p - 2, p - 1, p))))
                else:
                    new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p - 2, p - 1, p))))

            if p != 0 and (p - 1) not in e:
                if (p - cy) in e:
                    if (p + cy) not in e:
                        p2 = (p, p + 1, p + 2,
                              p + cx, p + cx + 1, p + cx + 2,
                              p + 2 * cx, p + 2 * cx + 1, p + 2 * cx + 2,
                              p + 3 * cx, p + 3 * cx + 1, p + 3 * cx + 2)
                        new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), p2)))
                    else:
                        new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p, p + 1, p + 2))))
                else:
                    new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p, p + 1, p + 2))))
            if (p - cy) not in e:
                new_e = np.append(new_e, list(filter(lambda point: point < (cx * cy), (p, p + cx, p + 2 * cx))))
    return new_e.astype('int64')
