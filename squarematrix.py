from copy import copy


class SquareMatrix:    
    def __init__(self, n):
        self.rows = [[0]*n for x in range(n)]
        self.n = n
        
    def __getitem__(self, i):
        return self.rows[i]

    def __setitem__(self, i, val):
        self.rows[i] = val
        
    def __str__(self):
        return str(self.rows)

    def __eq__(self, other):
        return (other.rows == self.rows)
        
    def __add__(self, other):
        result = SquareMatrix(self.n)
        
        for i in range(self.n):
            row = [val[0] + val[1] for val in zip(self.rows[i], other[i])]
            result[i] = row

        return result

    def __sub__(self, other):
        result = SquareMatrix(self.n)
        
        for i in range(self.n):
            row = [val[0]-val[1] for val in zip(self.rows[i], other[i])]
            result[i] = row

        return result
    
    def transposed(self):
        result = SquareMatrix.from_shape(self)
        result.rows = [list(item) for item in zip(*self.rows)]

        return result

    def __mul__(self, other):
        result = SquareMatrix.from_shape(n)
        other_t = other.transposed()
        
        for i in range(self.n):
            for j in range(other.n):
                result[i][j] = sum([val[0] * val[1] for val in zip(self.rows[i], other_t[j])])

        return result

    @classmethod
    def from_list(cls, rows_list):
        matrix = SquareMatrix(len(rows_list))
        matrix.rows = copy(rows_list)

        return matrix

    @classmethod
    def from_shape(cls, other):
        matrix = SquareMatrix(other.n)

        return matrix