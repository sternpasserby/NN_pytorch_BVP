from dataclasses import dataclass
import numpy as np

class FormattedTable:
    @dataclass
    class _ColumnInfo:
        name: str
        width: int
        format_str: str
        left_border: str
        right_border: str
        is_visible:bool = True

    @staticmethod
    def _verify_format_str(s: str) -> None:
        left = s.find('{')
        right = s.find('}')
        
        # 1. Check for exactly one '{' and one '}', and order
        if left == -1 or right == -1:
            raise ValueError("Format string must contain both '{' and '}' symbols.")
        if s.count('{') > 1 or s.count('}') > 1:
            raise ValueError("Format string must contain exactly one '{' and one '}'.")
        if left > right:
            raise ValueError("The '{' symbol must come before the '}' symbol.")
        
        # 2. Check for exactly one ':'
        if s.count(':') != 1:
            raise ValueError("Format string must contain exactly one ':' symbol.")
        
        colon = s.find(':')
        
        # 3. Check that ':' is between '{' and '}'
        if not (left < colon < right):
            raise ValueError("The ':' symbol must be inside the '{}' brackets.")

    def __init__(self, columns_info, n_rows=1):
        tmp = [None] * len(columns_info)
        for i in range(len(tmp)):
            # Название столбца
            name = columns_info[i][0]
            
            # Границы столбца
            s = columns_info[i][1]
            FormattedTable._verify_format_str(s)
            format_str = s
            left_border = s[:s.find('{')]
            right_border = s[s.find('}') + 1:]

            # Ширина столбца без его границ
            s = s[ s.find('{'): s.find('}') + 1 ]
            width = len(s.format(1))

            tmp[i] = FormattedTable._ColumnInfo(name, width, format_str, left_border, right_border)
        
        self.columns_info = tmp
        self.n_allocated_rows = n_rows
        self.n_rows = 0

        self.data = {col.name: np.empty(n_rows, dtype=object) for col in self.columns_info}

    def set_visibility(self, list_of_names: list[str], is_visible: bool):
        names = set(list_of_names)
        for column in self.columns_info:
            if column.name in names:
                column.is_visible = is_visible

    def _extend(self):
        data = self.data
        for key in data:
            arr = data[key]
            extra = np.empty_like(arr)
            data[key] = np.concatenate((arr, extra))
        self.n_allocated_rows *= 2

    def _header_as_string(self) -> str:
        s = ""
        c = ""
        for col in self.columns_info:
            if not col.is_visible:
                continue
            format_str = col.left_border + '{:>' + str(col.width) + 's}' + col.right_border
            s += format_str.format(col.name)

            c += col.left_border + '-' * col.width + col.right_border
        return s + '\n' + c.replace(" ", "-")
    
    def set_value(self, column_name: str, index: int, value: any) -> None:
        while index > self.n_allocated_rows - 1:
            self._extend()
        self.data[column_name][index] = value
        if self.n_rows < index + 1:
            self.n_rows = index + 1

    def row_as_string(self, i: int) -> str:
        if i + 1 > self.n_rows:
            ValueError(f"Row index out of bounds. You specified index {i:d}, but current table has maximum {self.n_rows} rows")
        data = self.data
        s = ''
        for col in self.columns_info:
            if not col.is_visible:
                continue
            x = data[col.name][i]
            if x is None: tmp = 'None'
            elif np.isnan(x): tmp = 'nan'
            elif np.isinf(x): tmp = 'inf' 
            else: tmp = x
            if x is None or np.isnan(x) or np.isinf(x):
                fmt = col.left_border + '{:>' + str(col.width) + 's}' + col.right_border
            else:
                fmt = col.format_str
            s += fmt.format(tmp)
        return s
    
    def __str__(self):
        s = self._header_as_string() + '\n'
        for i in range(self.n_rows):
            s += self.row_as_string(i) + '\n'
        return s