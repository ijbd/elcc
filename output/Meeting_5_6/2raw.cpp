#include <fstream>
using namespace std;

int main(){
    ifstream input("CISO_5_4.txt");
    ofstream output("CISO_raw.csv");
    string word;
    double elcc = 0;

    int year = 2016;
    while(input >> word){
        if(word == "ELCC:"){
            input >> elcc;
            output << year++ << ", " << elcc << endl;
        }
    }
    input.close();
    output.close();

    input.open("PACE_5_4.txt");
    output.open("PACE_raw.csv");
    elcc = 0;

    year = 2016;
    while(input >> word){
        if(word == "ELCC:"){
            input >> elcc;
            output << year++ << ", " << elcc << endl;
        }
    }
    input.close();
    output.close();

    input.open("WECC_5_5.txt");
    output.open("WECC_raw.csv");
    string location;
    while(input >> word){
        if(word == "Location:"){
            input >> location;
            output << location << ", ";
        }
        if(word == "ELCC:"){
            input >> elcc;
            output << elcc << endl;
        }
    }
    input.close();
    output.close();

    input.open("WECC_Nameplate_5_5.txt");
    output.open("WECC_Nameplate_raw.csv");
    string cap;
    while(input >> word){
        if(word == "Capacity:"){
            input >> cap;
            output << cap << ", ";
        }
        if(word == "ELCC:"){
            input >> elcc;
            output << elcc << endl;
        }
    }
    input.close();
    output.close();
    return 0;
}