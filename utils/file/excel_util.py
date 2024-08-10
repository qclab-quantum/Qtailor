
import pandas as pd
import  re

class ExcelUtil:
    @staticmethod
    def read_by_sheet(excel_file):
        # Load the Excel file
        xls = pd.ExcelFile(excel_file)

        # Get a list of all sheet names
        sheet_names = xls.sheet_names

        # Print the list of sheet names
        #print(sheet_names)

        data = {}
        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            data.update({sheet_name:df})
        #print(data['qnn'])
        return  sheet_names,data

def demo():
    # get data
    sheets,dfs = ExcelUtil.read_by_sheet('d:/temp.xlsx')
    # pharse data
    for sheet in sheets:
        df = dfs[sheet]
        circuits = df['circuit']
        #从字符串中提取出该线路的比特数量 qnn/qnn_indep_qiskit_5.qasm-> 5
        labels = list(map(lambda x: ''.join(re.findall(r'\d', x)), circuits))
        print(labels)

        rl = df['rl']
        qiskit = df['qiskit']
        mix= df['mix']

def row2array(sheet):
    data = []
    df = dfs[sheet]
    for index, row in df.iterrows():
        # print(f"Index: {index}")
        # print(f"Row:\n{row}\n")
        temp = []
        for header in ['ae_30',	'qnn_20',	'su2',	'portfvqe_15',	'qtf_20',	'pricingput_17',	'pricingcall_21']:
            temp.append(row[header])
        data.append(temp)
    print(data)


if __name__ == '__main__':
    sheets, dfs = ExcelUtil.read_by_sheet('d:/parallel.xlsx')
    qtailor = sheets[0]
    qiskit = sheets[1]
    row2array(qiskit)
    #row2array(qiskit)