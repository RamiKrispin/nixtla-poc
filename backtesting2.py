import pandas as pd


def partitions_mapping(input, index, partitions, overlap, test_length, train_length = None):
    
    df = None

    for i in range(partitions, 0, -1):
        if train_length is None:
            s = 1
        else:
            s = len(input) - train_length - i * test_length + overlap * (i -1) 

        e = len(input) - i * test_length  + overlap * (i -1) - 1
        train_start = input[index].iloc[s]
        train_end = input[index].iloc[e]
        test_start = input[index].iloc[e + 1]
        test_end = input[index].iloc[e + test_length]
        
        p_df = {"partition": partitions - i + 1,
                "train_start": [train_start], 
                "train_end" : [train_end],
                "test_start": [test_start], 
                "test_end" : [test_end],
                }
        if df is None:
            df = pd.DataFrame(p_df)
        else:
            temp = pd.DataFrame(p_df)
            df = df._append(temp)

    df = df.sort_values(by = ["partition"])
    df.reset_index(inplace = True, drop = True)
    return df

