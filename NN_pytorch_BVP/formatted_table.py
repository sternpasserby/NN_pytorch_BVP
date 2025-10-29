from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch

class FormattedTable:
    """
    A simple formatted table class for simple management of scientific computations

    Concept
    -------
    - The table data are stored in a dictionary of NumPy object arrays, where
      each key corresponds to a column name.
    - Column formatting metadata (width, format string, visibility, etc.) are stored
      in '_ColumnInfo' dataclass objects.
    - Each column's format string must contain exactly one pair of braces '{}' with
      a single ':' inside bracers and a format specifier. Everything before '{' and 
      after '}' are considered column's borders. Examples: "{:10d}" or "|{:>8.3f}|".
    """
    @dataclass
    class _ColumnInfo:
        """Metadata describing one column of the formatted table."""
        name: str
        width: int
        format_str: str
        left_border: str
        right_border: str
        is_visible: bool = True

    @staticmethod
    def _verify_format_str(s: str) -> None:
        """
        Validate a format string containing a single placeholder.

        Requirements
        ------------
        - Must contain exactly one opening and one closing brace.
        - Must contain exactly one colon ':' located between the braces.

        Notes
        -----
        This is a simplified syntax checker that does not support nested
        or escaped braces. For example, "{:>10.3f}" or "|{:8s}|" are valid.
        """
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

    @staticmethod
    def _extract_width(fmt: str) -> int | None:
        """
        Attempt to extract the width field from a format specifier.

        Example
        -------
        fmt="*>10.2f" → returns 10

        Parameters
        ----------
        fmt : str
            The substring following the colon in a format string, e.g. "10.3f".

        Returns
        -------
        width : int or None
            Extracted width if found, otherwise None.
        """
        i = 0
        n = len(fmt)

        # Step 1: Skip optional fill + alignment
        if n >= 2 and fmt[1] in "<>^=":
            i = 2  # skip fill + alignment
        elif n >= 1 and fmt[0] in "<>^=":
            i = 1  # skip alignment only


        # Step 2: Skip optional sign
        if i < n and fmt[i] in "+- ":
            i += 1

        # Step 3: Extract width (sequence of digits)
        width_str = ""
        while i < n and fmt[i].isdigit():
            width_str += fmt[i]
            i += 1

        return int(width_str) if width_str else None

    def __init__(self, columns_info, n_rows=1):
        """
        Initialize the table with column definitions.

        Parameters
        ----------
        columns_info : sequence of (str, str)
            A sequence of tuples (name, format_string) defining each column.
            Example: [("x", "{:10.3f}"), ("label", "{:>8s}")].
        n_rows : int, optional
            Initial number of allocated rows (default is 1). The table will
            automatically expand as needed.
        """
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
            width = FormattedTable._extract_width(s[ s.find(':') + 1: s.find('}') ])
            if width is None:
                raise ValueError(f"Could not extract width from a format specifier {s}")

            tmp[i] = FormattedTable._ColumnInfo(name, width, format_str, left_border, right_border)
        
        self.columns_info = tmp
        self.n_allocated_rows = n_rows
        self.n_rows = 0

        self.data = {col.name: np.empty(n_rows, dtype=object) for col in self.columns_info}

    def set_visibility(self, list_of_names: list[str], is_visible: bool):
        """
        Change the visibility of specific columns.

        Parameters
        ----------
        list_of_names : iterable of str
            Names of columns to modify.
        is_visible : bool
            Whether to show (True) or hide (False) the specified columns.
        """
        names = set(list_of_names)
        for column in self.columns_info:
            if column.name in names:
                column.is_visible = is_visible

    def _extend(self):
        """
        Double the allocated row capacity of each column.
        """
        data = self.data
        for key in data:
            arr = data[key]
            extra = np.empty_like(arr)
            data[key] = np.concatenate((arr, extra))
        self.n_allocated_rows *= 2

    def _header_as_string(self) -> str:
        """
        Return the table header consisting of column names and separators.

        Returns
        -------
        header : str
            Two-line string containing the column names and a line of dashes.
        """
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
        """
        Assign a value to a specific table cell.

        Parameters
        ----------
        column_name : str
            Name of the column (must exist in the table).
        index : int
            Zero-based row index. The table will automatically expand if needed.
        value : any
            Value to store. All entries are stored as Python objects.
        """
        while index > self.n_allocated_rows - 1:
            self._extend()
        self.data[column_name][index] = value
        if self.n_rows < index + 1:
            self.n_rows = index + 1

    def row_as_string(self, i: int) -> str:
        """
        Format a single row into a printable string.

        Parameters
        ----------
        i : int
            Zero-based row index.

        Returns
        -------
        row : str
            The formatted line representing row `i`.

        Raises
        ------
        ValueError
            If the index is greater than or equal to the number of rows.
        """
        if i + 1 > self.n_rows:
            raise ValueError(f"Row index out of bounds. You specified index {i:d}, but current table has maximum {self.n_rows} rows")
        data = self.data
        s = ''
        for col in self.columns_info:
            if not col.is_visible:
                continue

            x = data[col.name][i]
            if isinstance(x, (int, float, complex, np.number, torch.Tensor)):
                fmt = col.format_str
                s += fmt.format(x)
            else:
                fmt = col.left_border + '{:>' + str(col.width) + 's}' + col.right_border
                s += fmt.format(str(x))

        return s
    
    def __str__(self):
        s = self._header_as_string() + '\n'
        for i in range(self.n_rows):
            s += self.row_as_string(i) + '\n'
        return s
    
    def __eq__(self, other):
        if isinstance(other, FormattedTable):
            if self.n_allocated_rows != other.n_allocated_rows:
                return False
            if self.n_rows != other.n_rows:
                return False

            # Compare columnts info
            if len(self.columns_info) == len(other.columns_info):
                for i in range(len(self.columns_info)):
                    if self.columns_info[i] == other.columns_info[i]:
                        continue
                    else:
                        return False
            else:
                return False
            
            # Compare data
            for key in self.data:
                if np.array_equal(self.data[key], other.data[key]):
                    continue
                else:
                    return False
            return True
        return False

    @classmethod
    def save(cls, table, path: Path):
        """
        Serialize the table to a binary file using pickle.

        Parameters
        ----------
        table : FormattedTable
            Table instance to save.
        path : pathlib.Path
            Path to the output file.
        """
        with open(path, "wb") as outfile:
            pickle.dump(table, outfile)

    @classmethod
    def load(cls, path: Path):
        """
        Load a table from a pickle file.

        Parameters
        ----------
        path : pathlib.Path
            Path to the pickle file.

        Returns
        -------
        table : FormattedTable
            The deserialized table instance.
        """
        with open(path, "rb") as infile:
            table = pickle.load(infile)
        return table

    
if __name__ == "__main__":
    table1 = FormattedTable([
        ("column_str", "{:10s} |"),
        ("column_decimal", "{:20d}"),
        ("column_float", "{:15.2f}"),
        ("column_bool", "{:15}"),
        ("column_torch", "{:15.2f}")])
    table1.set_value("column_str", 0, "line")
    table1.set_value("column_float", 2, 3.3)
    table1.set_value("column_decimal", 0, 137)
    table1.set_value("column_decimal", 1, 1)
    table1.set_value("column_bool", 5, True)
    table1.set_value("column_bool", 6, False)
    table1.set_value("column_torch", 3, torch.tensor(3.3))
    print(table1)

    temp_path = Path.cwd() / 'tests' / 'temp'
    temp_path.mkdir(exist_ok=True)

    FormattedTable.save(table1, temp_path / "table1.pickle")
    table2 = FormattedTable.load(temp_path / "table1.pickle")
    print(table2)
    
    assert table1 == table2
    table2.set_value("column_decimal", 1, 3)
    assert table1 != table2

    assert FormattedTable._extract_width("*<-012,.4f") == 12
    assert FormattedTable._extract_width("<-10,.4f") == 10
    assert FormattedTable._extract_width("<4,.4f") == 4
    assert FormattedTable._extract_width("5.4f") == 5
    assert FormattedTable._extract_width("013d") == 13
    assert "tensor" not in table1.row_as_string(3)
    
    
"""
Python f-string format specifier cheat sheet

Example:
    f"{x:*<+010,.2f}"

Components:
    x       : any Python expression
    :       : separates expression from format specifier

    Fill & Alignment:
        *<    : fill character (*) and alignment (< left, > right, ^ center)
        *     : optional, default is space

    Sign:
        +     : always show sign
        -     : show only if negative
        space : show space if positive

    Width & Zero-padding:
        10    : minimum width
        010   : width 10, padded with zeros (optional leading 0)

    Thousands separator:
        ,     : comma
        _     : underscore

    Precision:
        .2    : digits after decimal (for floats)

    Type:
        d     : decimal integer
        o     : octal
        x/X   : hexadecimal
        f     : float
        e/E   : exponential
        %     : percentage

Full Example:
    x = 1234.567
    f"{x:*<+010,.2f}"  # -> '+1234.57***'
"""
