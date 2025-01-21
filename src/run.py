import SoilingNumberPredictionModel as snpm
import sys

def main():
    # if len(sys.argv) not in [1, 2]:
    #     sys.exit("Usage: C:/Users\sruth\Downloads\pvdaq_system_4_2010-2016_subset_soil_signal.csv")
    snpm.prediction_model("C:/Users/sruth/Downloads/pvdaq_system_4_2010-2016_subset_soil_signal.csv")

if __name__ == "__main__":
    main()