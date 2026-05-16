
import pandas as pd
import sys
from sdv.sequential import PARSynthesizer
from sdv.metadata import SingleTableMetadata



def main (fileName):
    
    df = pd.read_csv(fileName, sep=";", names=["Date", "column"], header=0)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.reset_index(drop=True)
    df.insert(0, "id", df.index)
    df = df.sort_values("Date").reset_index(drop=True)

    all_months = pd.date_range(df["Date"].min(), df["Date"].max(), freq="MS")
    df_full = pd.DataFrame({"Date": all_months})

    df_merged = df_full.merge(df, on="Date", how="left")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df[["Date", "column"]])
    metadata.add_column(column_name="id", sdtype="id")
    metadata.set_sequence_key("id") 

    synth = PARSynthesizer(metadata)
    synth.fit(df)

    n_missing = df_merged["column"].isna().sum()
    synthetic = synth.sample(num_sequences=1, sequence_length=n_missing)

    df_merged.loc[df_merged["column"].isna(), "column"] = synthetic["column"].values
    df_merged = df_merged.drop('id', axis=1)
    df_merged["Date"] = df_merged["Date"].dt.strftime("%d-%m-%Y")

    print(df_merged)
    df_merged.to_csv("Oil_synthetic.csv", index=False, sep=";")



if __name__ == "__main__":
   
    if len(sys.argv) < 1:
         
         print(f"Error, forcing reading default data")
         fileName1="Oil_old.csv"
    else:
        fileName1=sys.argv[1]
    
    main(fileName1)
