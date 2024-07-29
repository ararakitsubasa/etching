from numba import jit
import numpy as np


react_table = np.array([[[0.01, 2], [0.01, 3], [0.01, 4], [0.01, -4], [0.05, 7], [0.00, 0], [0.05, 8], [0.00, 0], [0.06, 10], [0.00, 0]],
                        [[0.05, 5], [0.00, 0], [0.00, 0], [0.00, 0], [0.05, 6], [0.00, 0], [0.00, 0], [0.00, 0], [0.00, 0], [0.00, 0]],
                        [[0.27, -1], [0.27, -2], [0.27, -3], [0.27, -4], [0.27, -5], [0.27, -6], [0.27, -7], [0.27, -8], [0.27, -9], [0.27, -10]]])

@jit(nopython=True)
def reaction(parcel, film):
    num_parcels = parcel.shape[0]
    num_reactions = react_table.shape[1]
    choice = np.random.rand(num_parcels, num_reactions)
    parcelGen = np.zeros(num_parcels)
    reactList = np.zeros(num_parcels)

    for i in range(num_parcels):
        acceptList = react_table[parcel[i], :, 0] > choice[i]
        react_choice_indices = np.where(acceptList)[0]
        # print(react_choice_indices)
        if react_choice_indices.size > 0:
            react_choice = np.random.choice(react_choice_indices)
            reactList[i] = react_choice
            film[i, react_choice] -= 0.01
            react_gen = react_table[parcel[i], react_choice, 1]
            if react_gen > 0:
                film[i, int(react_gen) - 1] += 0.01
            else:
                parcelGen[i] = -react_gen
    
    return film, parcelGen, reactList

class UnitTest:
    def __init__(self, parcel, film):
        self.parcel = parcel
        self.film = film
    
    def testReact(self):
        film, parcelGen, reactList = reaction(self.parcel, self.film)
        return film, parcelGen, reactList