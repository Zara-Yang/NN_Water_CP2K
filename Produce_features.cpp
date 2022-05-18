#include <vector>
#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <omp.h>

using namespace std;

typedef vector<vector<double>> MAT2D;  
typedef vector<MAT2D> MAT3D;
typedef vector<MAT3D> MAT4D;
typedef vector<double> VEC; 

#define natoms 192      // the total number of nucleus
#define noxygen 64      // the number of oxygen
#define nhydrogen 128   // the number of hydrogen
#define nwannier 256
#define R_c 6           // R_cut Ans
ostream& operator<<(ostream& out, const MAT2D& v) {
    for (int x = 0; x < v.size(); x++) {
        for (int y = 0; y < v[0].size(); y++) {
            out << v[x][y] << " ";}
        cout << endl;}
    return(out);}

double fc(double r)
{
    double rc = R_c;
    double y = 0;
    if (r < rc)
    {
        y = pow(tanh(1 - r / rc), 3);
    }
    return y;
}
double dfc(double r)
{
    double rc = R_c;
    double y = 0;
    if (r < rc)
    {
        y = -3 * pow(tanh(1 - r / rc), 2) / pow(cosh(1 - r / rc), 2) / rc;
    }
    return y;
}
double G2(double r12, double yeta, double rs)
{
    double y = exp(-yeta * (r12 - rs) * (r12 - rs)) * fc(r12);
    return y;
}
double dG2(double r12, double yeta, double rs) // this is only the radial part of dG2
{
    double y = -2 * yeta * (r12 - rs) * fc(r12) * exp(-yeta * pow((r12 - rs), 2)) + exp(-yeta * pow((r12 - rs), 2)) * dfc(r12);
    return y;
}
double G4(double r12, double r13, double r23, double cosalphaijk, double zeta, double yeta, double lam)
{
    double y = exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * fc(r12) * fc(r13) * fc(r23) * pow((1 + lam * cosalphaijk), zeta);
    y = y * pow(2, 1 - zeta);
    return y;
}
VEC dG4_ij_ik_jk(double r12,double r13,double r23, double cosalphaijk, double zeta, double yeta, double lam){
    double fc12 = fc(r12);
    double fc13 = fc(r13);
    double fc23 = fc(r23);
    double dfc12 = dfc(r12);
    double dfc13 = dfc(r13);
    double dfc23 = dfc(r23);
    
    double commonvalue = pow(2, 1 - zeta) * exp(-yeta * (r12 * r12 + r13 * r13 + r23 * r23)) * pow((1 + lam * cosalphaijk), (zeta-1)) ;
    double y = 0;
    y += zeta * lam * ( 1.0/ r13 - (r12*r12 + r13*r13 - r23*r23) / (2*r12*r12*r13))  * fc12 * fc13 *fc23;
    y += - 2 * r12 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
    y += dfc12 * (1 + lam * cosalphaijk) * fc13 *fc23;
    y = y * commonvalue;
    
    double y1 = 0;
    y1 += zeta * lam * ( 1.0/ r12 - (r12*r12 + r13*r13 - r23*r23) / (2*r13*r13*r12))  * fc12 * fc13 *fc23;
    y1 += - 2 * r13 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
    y1 += dfc13 * (1 + lam * cosalphaijk) * fc12 *fc23;
    y1 = y1 * commonvalue;
    
    double y2 = 0;
    y2 += zeta * lam * (-r23/(r12 * r13)) * fc12 * fc13 *fc23;
    y2 += - 2 * r23 * yeta  * (1 + lam * cosalphaijk) * fc12 * fc13 *fc23;
    y2 += dfc23 * (1 + lam * cosalphaijk) * fc12 *fc13;
    y2 = y2 * commonvalue;
    
    VEC ys(3);
    ys[0] = y;
    ys[1] = y1;
    ys[2] = y2;
    return ys;
}

void read_parameters(MAT2D& parameters, string fp_name, int nx)
{
    ifstream fp(fp_name);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            fp >> parameters[i][j];
        }
    }
}

void get_G2features(MAT2D& features, MAT2D& params, MAT2D& Norm_r, int id_i, int id_j, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            
            if ((j != i) && Norm_r[i][j] < R_c)
            {
                for (int ip = 0; ip < params.size(); ip++)
                {
                    features[ip][i - id_i * noxygen] += G2(Norm_r[i][j], params[ip][1], params[ip][0]);
                }
            }
        }
    }
    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - id_i * noxygen] << " ";
        }
        fp << "\n";
    }
    fp.close();
}

void get_WG2features(MAT2D& features, MAT2D& params, MAT2D& Norm_r, int id_i, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0,nwannier};

    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            
            if ((j != i) && Norm_r[i][j] < R_c)
            {
                for (int ip = 0; ip < params.size(); ip++)
                {
                    features[ip][i - id_i * noxygen] += G2(Norm_r[i][j], params[ip][1], params[ip][0]);
                }
            }
        }
    }
    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - id_i * noxygen] << " ";
        }
        fp << "\n";
    }
    fp.close();

}

void get_G4features(MAT2D& features, MAT2D& params, MAT2D& Norm_r, int id_i, int id_j, int id_k, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    int rangek[2] = {0, noxygen};
    if (id_k == 1)
    {
        rangek[0] = noxygen;
        rangek[1] = natoms;
    }
    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && Norm_r[i][j] < R_c)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i) && Norm_r[i][k] < R_c && Norm_r[j][k] < R_c)
                    {
                        double cosijk = (Norm_r[i][j] * Norm_r[i][j] + Norm_r[i][k] * Norm_r[i][k] - Norm_r[j][k] * Norm_r[j][k]) / (2 * Norm_r[i][j] * Norm_r[i][k]);
                        for (int ip = 0; ip < params.size(); ip++)
                        {
                            features[ip][i - id_i * noxygen] += G4(Norm_r[i][j], Norm_r[i][k], Norm_r[j][k], cosijk, params[ip][3], params[ip][1], params[ip][2]);
                        }
                    }
                }
            }
        }
    }

    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - id_i * noxygen] << " ";
        }
        fp << "\n";
    }
    fp.close();
}

void get_WG4features(MAT2D& features, MAT2D& params, MAT2D& Norm_nn_r, MAT2D& Norm_nw_r, int id_i, int id_j, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    int rangek[2] = {0, nwannier};

    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && Norm_nn_r[i][j] < R_c)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i) && Norm_nw_r[i][k] < R_c && Norm_nw_r[j][k] < R_c)
                    {
                        double cosijk = (Norm_nn_r[i][j] * Norm_nn_r[i][j] + Norm_nw_r[i][k] * Norm_nw_r[i][k] - Norm_nw_r[j][k] * Norm_nw_r[j][k]) / (2 * Norm_nn_r[i][j] * Norm_nw_r[i][k]);
                        for (int ip = 0; ip < params.size(); ip++)
                        {
                            features[ip][i - id_i * noxygen] += G4(Norm_nn_r[i][j], Norm_nw_r[i][k], Norm_nw_r[j][k], cosijk, params[ip][3], params[ip][1], params[ip][2]);
                        }
                    }
                }
            }
        }
    }

    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            fp << features[ip][i - id_i * noxygen] << " ";
        }
        fp << "\n";
    }
    fp.close();
}

void get_dG2features(MAT4D& features, MAT2D& params, MAT2D& Norm_r, MAT3D& Vec_r, int id_i, int id_j, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && Norm_r[i][j] < R_c)
            {
                for (int ip = 0; ip < params.size(); ip++)
                {
                    double dG2ij = dG2(Norm_r[i][j], params[ip][1], params[ip][0]);
                    for (int ix = 0; ix < 3; ix++)
                    {
                        features[ip][i - id_i * noxygen][j][ix] += dG2ij * Vec_r[i][j][ix] / Norm_r[i][j];
                        
                        features[ip][i - id_i * noxygen][i][ix] -= dG2ij * Vec_r[i][j][ix] / Norm_r[i][j];
                        
                    }
                }
            }
        }
    }
     
    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - id_i * noxygen][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

void get_dWG2features(MAT4D& features, MAT2D& params, MAT2D& Norm_r, MAT3D& Vec_r, int id_i, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0,nwannier};
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && Norm_r[i][j] < R_c)
            {
                for (int ip = 0; ip < params.size(); ip++)
                {
                    double dG2ij = dG2(Norm_r[i][j], params[ip][1], params[ip][0]);
                    for (int ix = 0; ix < 3; ix++)
                    {                        
                        features[ip][i - id_i * noxygen][i][ix] -= dG2ij * Vec_r[i][j][ix] / Norm_r[i][j];
                    }
                }
            }
        }
    }
    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - id_i * noxygen][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

void get_dG4features(MAT4D& features, MAT2D& params, MAT2D& Norm_r, MAT3D& Vec_r, int id_i, int id_j, int id_k, string fname)
{
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    int rangek[2] = {0, noxygen};
    if (id_k == 1)
    {
        rangek[0] = noxygen;
        rangek[1] = natoms;
    }
    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && Norm_r[i][j] < R_c)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i)  && Norm_r[i][k] < R_c && Norm_r[j][k] < R_c)
                    {
                        double cosijk = (Norm_r[i][j] * Norm_r[i][j] + Norm_r[i][k] * Norm_r[i][k] - Norm_r[j][k] * Norm_r[j][k]) / (2 * Norm_r[i][j] * Norm_r[i][k]);
                        for (int ip = 0; ip < params.size(); ip++)
                        {
                            vector<double> Gs = dG4_ij_ik_jk(Norm_r[i][j], Norm_r[i][k], Norm_r[j][k], cosijk, params[ip][3], params[ip][1], params[ip][2]);
                            double dG4_ij_ij = Gs[0];
                            double dG4_ij_ik = Gs[1];
                            double dG4_jk_jk = Gs[2];
                            for (int ix = 0; ix < 3; ix++)
                            {
                                features[ip][i - id_i * noxygen][j][ix] += dG4_ij_ij * Vec_r[i][j][ix] / Norm_r[i][j];
                                features[ip][i - id_i * noxygen][j][ix] += dG4_jk_jk * Vec_r[k][j][ix] / Norm_r[j][k];
                                
                                features[ip][i - id_i * noxygen][k][ix] += dG4_ij_ik * Vec_r[i][k][ix] / Norm_r[i][k];
                                features[ip][i - id_i * noxygen][k][ix] += dG4_jk_jk * Vec_r[j][k][ix] / Norm_r[k][j];
                                
                                features[ip][i - id_i * noxygen][i][ix] -= dG4_ij_ij * Vec_r[i][j][ix] / Norm_r[i][j];
                                features[ip][i - id_i * noxygen][i][ix] -= dG4_ij_ik * Vec_r[i][k][ix] / Norm_r[i][k];
                            }
                        }
                    }
                }
            }
        }
    }
    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - id_i * noxygen][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

void get_dWG4features(MAT4D& features, MAT2D& params, MAT2D& Norm_nn_r, MAT3D& Vec_nn_r, MAT2D& Norm_nw_r, MAT3D& Vec_nw_r,int id_i, int id_j, string fname){
    int rangei[2] = {0, noxygen};
    if (id_i == 1)
    {
        rangei[0] = noxygen;
        rangei[1] = natoms;
    }
    int rangej[2] = {0, noxygen};
    if (id_j == 1)
    {
        rangej[0] = noxygen;
        rangej[1] = natoms;
    }
    int rangek[2] = {0, nwannier};

    #pragma omp parallel for  
    for (int i = rangei[0]; i < rangei[1]; i++)
    {
        for (int j = rangej[0]; j < rangej[1]; j++)
        {
            if ((j != i) && Norm_nn_r[i][j] < R_c)
            {
                for (int k = rangek[0]; k < rangek[1]; k++)
                {
                    if ((k != j) && (k != i)  && Norm_nw_r[i][k] < R_c && Norm_nw_r[j][k] < R_c)
                    {
                        double cosijk = (Norm_nn_r[i][j] * Norm_nn_r[i][j] + Norm_nw_r[i][k] * Norm_nw_r[i][k] - Norm_nw_r[j][k] * Norm_nw_r[j][k]) / (2 * Norm_nw_r[i][j] * Norm_nw_r[i][k]);
                        for (int ip = 0; ip < params.size(); ip++)
                        {
                            vector<double> Gs = dG4_ij_ik_jk(Norm_nn_r[i][j], Norm_nw_r[i][k], Norm_nw_r[j][k], cosijk, params[ip][3], params[ip][1], params[ip][2]);
                            double dG4_ij_ij = Gs[0];
                            double dG4_ij_ik = Gs[1];
                            double dG4_jk_jk = Gs[2];
                            for (int ix = 0; ix < 3; ix++)
                            {
                                features[ip][i - id_i * noxygen][j][ix] += dG4_ij_ij * Vec_nn_r[i][j][ix] / Norm_nn_r[i][j];
                                features[ip][i - id_i * noxygen][j][ix] += dG4_jk_jk * Vec_nw_r[j][k][ix] / Norm_nw_r[j][k];
                                
                                // features[ip][i - id_i * noxygen][k][ix] += dG4_ij_ik * Vec_r[i][k][ix] / Norm_r[i][k];
                                // features[ip][i - id_i * noxygen][k][ix] += dG4_jk_jk * Vec_r[j][k][ix] / Norm_r[k][j];
                                
                                features[ip][i - id_i * noxygen][i][ix] -= dG4_ij_ij * Vec_nn_r[i][j][ix] / Norm_nn_r[i][j];
                                features[ip][i - id_i * noxygen][i][ix] -= dG4_ij_ik * Vec_nw_r[i][k][ix] / Norm_nw_r[i][k];
                            }
                        }
                    }
                }
            }
        }
    }
    ofstream fp(fname);
    for (int ip = 0; ip < params.size(); ip++)
    {
        for (int i = rangei[0]; i < rangei[1]; i++)
        {
            for (int j = 0; j < natoms; j++)
            {
                for (int ix = 0; ix < 3; ix++)
                {
                    fp << features[ip][i - id_i * noxygen][j][ix] << " ";
                }
            }
        }
    }
    fp << "\n";
    fp.close();
}

void Load_box_range(VEC& box_range,string box_range_path){
    ifstream fbox(box_range_path);
    for (int i =0;i < 3;i++){
        fbox >> box_range[i]; // read in the boxlength of this configuration
    }
    fbox.close();
}

void Load_Coord(MAT2D& mat,string fO_coord_path,string fH_coord_path){
    ifstream fOxyz(fO_coord_path);
    for (int i = 0; i < noxygen; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fOxyz >> mat[i][j];
            //cout << nxyz[i][j] << "  ";
        }
        //cout << "\n";
    }
    fOxyz.close();

    ifstream fHxyz(fH_coord_path);
    for (int i = noxygen; i < (nhydrogen + noxygen); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            fHxyz >> mat[i][j];
            //cout << nxyz[i][j] << "  ";
        }
        //cout << "\n";
    }
    fHxyz.close();
}

void Load_wannier_coord(MAT2D& mat,string wannier_path){
    ifstream Wxyz(wannier_path);
    for (int i = 0; i < nwannier; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            Wxyz >> mat[i][j];
            //cout << nxyz[i][j] << "  ";
        }
        //cout << "\n";
    }
    Wxyz.close();
}

void Compute_norm_vec(MAT2D& rij,MAT3D& vecrij, MAT2D& nxyz,VEC& boxlength){
    for (int i = 0; i < natoms; i++)
    { //initialize rij
        for (int j = 0; j < natoms; j++)
        {
            rij[i][j] = 0;
        }
    }

    // get rij and vecrij
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < natoms; j++)
        {
            double disx = nxyz[j][0] - nxyz[i][0] - round((nxyz[j][0] - nxyz[i][0]) / boxlength[0]) * boxlength[0];
            double disy = nxyz[j][1] - nxyz[i][1] - round((nxyz[j][1] - nxyz[i][1]) / boxlength[1]) * boxlength[1];
            double disz = nxyz[j][2] - nxyz[i][2] - round((nxyz[j][2] - nxyz[i][2]) / boxlength[2]) * boxlength[2];
            // cout << disx << ";" << disy << ";" << disz << endl;
            double dis = sqrt(disx * disx + disy * disy + disz * disz);
            rij[i][j] = dis;
            vecrij[i][j][0] = disx;
            vecrij[i][j][1] = disy;
            vecrij[i][j][2] = disz;
            // cout << vecrij[i][j][0] << "  " << vecrij[i][j][1]  << " " << vecrij[i][j][2] << " ; " << rij[i][j];
        }
        // cout << "\n";
    }
}

void Compute_norm_nv_vec(MAT2D& nw_rij,MAT3D& vec_nw_rij, MAT2D& nxyz, MAT2D& wxyz, VEC& boxlength){
    for (int i = 0; i < natoms; i++)
    { //initialize rij
        for (int j = 0; j < nwannier; j++)
        {
            nw_rij[i][j] = 0;
            for (int dim = 0; dim < 3;dim ++){
                vec_nw_rij[i][j][dim] = 0;
            }
        }
    }

    // get rij and vecrij
    for (int i = 0; i < natoms; i++)
    {
        for (int j = 0; j < nwannier; j++)
        {
            double disx = wxyz[j][0] - nxyz[i][0] - round((wxyz[j][0] - nxyz[i][0]) / boxlength[0]) * boxlength[0];
            double disy = wxyz[j][1] - nxyz[i][1] - round((wxyz[j][1] - nxyz[i][1]) / boxlength[1]) * boxlength[1];
            double disz = wxyz[j][2] - nxyz[i][2] - round((wxyz[j][2] - nxyz[i][2]) / boxlength[2]) * boxlength[2];
            double dis = sqrt(disx * disx + disy * disy + disz * disz);
            nw_rij[i][j] = dis;
            vec_nw_rij[i][j][0] = disx;
            vec_nw_rij[i][j][1] = disy;
            vec_nw_rij[i][j][2] = disz;
        }
    }
}

int main(){
    omp_set_num_threads(20);
    // Matrix variables
    MAT2D nxyz(natoms, vector<double>(3));
    MAT2D wxyz(nwannier,vector<double>(3));
    MAT2D rij(natoms, vector<double>(natoms));       // get the distance matrix for the nucleus
    MAT2D nw_rij(natoms,vector<double>(nwannier));
    MAT3D vecrij(natoms, MAT2D(natoms, vector<double>(3))); // the vector between i and j
    MAT3D vec_nw_rij(natoms,MAT2D(nwannier, vector<double>(3)));
    VEC boxlength = {0,0,0};

    MAT2D parameters2_OO(8, vector<double>(4));
    MAT2D parameters2_OH(8, vector<double>(4));
    MAT2D parameters2_HO(8, vector<double>(4));
    MAT2D parameters2_HH(8, vector<double>(4));

    MAT2D parameters4_OOO(4, vector<double>(4));
    MAT2D parameters4_OOH(4, vector<double>(4));
    MAT2D parameters4_OHH(6, vector<double>(4));
    MAT2D parameters4_HHO(4, vector<double>(4));
    MAT2D parameters4_HOO(4, vector<double>(4));

    MAT2D features_G2OO(8, vector<double>(noxygen));
    MAT2D features_G2OH(8, vector<double>(noxygen));
    MAT2D features_G2HO(8, vector<double>(nhydrogen));
    MAT2D features_G2HH(8, vector<double>(nhydrogen));

    MAT2D features_G2OW(8, vector<double>(noxygen));
    MAT2D features_G2HW(8, vector<double>(nhydrogen));

    MAT2D features_G4OOO(4, vector<double>(noxygen));
    MAT2D features_G4OOH(4, vector<double>(noxygen));
    MAT2D features_G4OHH(6, vector<double>(noxygen));
    MAT2D features_G4HHO(4, vector<double>(nhydrogen));
    MAT2D features_G4HOO(4, vector<double>(nhydrogen));

    MAT2D features_G4OHW(4, vector<double>(noxygen));
    MAT2D features_G4HOW(4, vector<double>(nhydrogen));

    MAT4D features_dG2OO(8, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG2OH(8, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG2HO(8, MAT3D(nhydrogen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG2HH(8, MAT3D(nhydrogen, MAT2D(natoms, vector<double>(3))));
    
    MAT4D features_dG2OW(8, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG2HW(8, MAT3D(nhydrogen, MAT2D(natoms, vector<double>(3))));

    MAT4D features_dG4OHW(4, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG4HOW(4, MAT3D(nhydrogen, MAT2D(natoms, vector<double>(3))));

    MAT4D features_dG4OOO(4, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG4OOH(4, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG4OHH(6, MAT3D(noxygen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG4HHO(4, MAT3D(nhydrogen, MAT2D(natoms, vector<double>(3))));
    MAT4D features_dG4HOO(4, MAT3D(nhydrogen, MAT2D(natoms, vector<double>(3))));
    read_parameters(parameters2_OO, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G2_parameters_OO.txt", 8);
    read_parameters(parameters2_OH, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G2_parameters_OH.txt", 8);
    read_parameters(parameters2_HO, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G2_parameters_HO.txt", 8);
    read_parameters(parameters2_HH, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G2_parameters_HH.txt", 8);

    read_parameters(parameters4_OOO, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G4_parameters_OOO.txt", 4);
    read_parameters(parameters4_OOH, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G4_parameters_OOH.txt", 4);
    read_parameters(parameters4_OHH, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G4_parameters_OHH.txt", 6);
    read_parameters(parameters4_HHO, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G4_parameters_HHO.txt", 4);
    read_parameters(parameters4_HOO, "/DATA/users/yanghe/projects/Behler_Parrinello_Network/BPNN_project/Code/parameters/G4_parameters_HOO.txt", 4);
    /* 
    cout << parameters2_OO << endl;
    cout << parameters2_OH << endl;
    cout << parameters2_HO << endl;
    cout << parameters2_HH << endl;

    cout << parameters4_OOO << endl;
    cout << parameters4_OOH << endl;
    cout << parameters4_OHH << endl;
    cout << parameters4_HHO << endl;
    cout << parameters4_HOO << endl;
    */
    Load_box_range(boxlength,"Box.txt");
    Load_Coord(nxyz,"Oxyz.txt","Hxyz.txt");
    Load_wannier_coord(wxyz,"Wxyz.txt");
    // cout << "Box length : " << boxlength[0] << "-" << boxlength[1] << "-" << boxlength[2] << endl;

    Compute_norm_vec(rij,vecrij,nxyz,boxlength);

    Compute_norm_nv_vec(nw_rij,vec_nw_rij,nxyz,wxyz,boxlength);

    // get G2 features
    get_G2features(features_G2OO, parameters2_OO, rij, 0, 0, "./features/features_G2OO.txt");
    get_G2features(features_G2OH, parameters2_OH, rij, 0, 1, "./features/features_G2OH.txt");
    get_G2features(features_G2HO, parameters2_HO, rij, 1, 0, "./features/features_G2HO.txt");
    get_G2features(features_G2HH, parameters2_HH, rij, 1, 1, "./features/features_G2HH.txt");

    // get WG2 features
    get_WG2features(features_G2OW,parameters2_OO, nw_rij, 0, "./features/features_G2OW.txt");
    get_WG2features(features_G2HW,parameters2_HH, nw_rij, 1, "./features/features_G2HW.txt");

    // get G4 features
    get_G4features(features_G4OOO, parameters4_OOO, rij, 0, 0, 0, "./features/features_G4OOO.txt");
    get_G4features(features_G4OOH, parameters4_OOH, rij, 0, 0, 1, "./features/features_G4OOH.txt");
    get_G4features(features_G4OHH, parameters4_OHH, rij, 0, 1, 1, "./features/features_G4OHH.txt");
    get_G4features(features_G4HHO, parameters4_HHO, rij, 1, 1, 0, "./features/features_G4HHO.txt");
    get_G4features(features_G4HOO, parameters4_HOO, rij, 1, 0, 0, "./features/features_G4HOO.txt");

    // get WG4 features
    get_WG4features(features_G4OHW,parameters4_OOH,rij,nw_rij,0,1,"./features/features_G4OHW.txt");
    get_WG4features(features_G4HOW,parameters4_HOO,rij,nw_rij,1,0,"./features/features_G4HOW.txt");

    // get dG2 features
    get_dG2features(features_dG2OO, parameters2_OO, rij, vecrij, 0, 0, "./features/features_dG2OO.txt");
    get_dG2features(features_dG2OH, parameters2_OH, rij, vecrij, 0, 1, "./features/features_dG2OH.txt");
    get_dG2features(features_dG2HO, parameters2_HO, rij, vecrij, 1, 0, "./features/features_dG2HO.txt");
    get_dG2features(features_dG2HH, parameters2_HH, rij, vecrij, 1, 1, "./features/features_dG2HH.txt");

    // get dWG2 features
    get_dWG2features(features_dG2OW, parameters2_OO, nw_rij, vec_nw_rij, 0, "./features/features_dG2OW.txt");
    get_dWG2features(features_dG2HW, parameters2_HH, nw_rij, vec_nw_rij, 1, "./features/features_dG2HW.txt");
    
    // get dG4 features
    get_dG4features(features_dG4OOO, parameters4_OOO, rij, vecrij, 0, 0, 0, "./features/features_dG4OOO.txt");
    get_dG4features(features_dG4OOH, parameters4_OOH, rij, vecrij, 0, 0, 1, "./features/features_dG4OOH.txt");
    get_dG4features(features_dG4OHH, parameters4_OHH, rij, vecrij, 0, 1, 1, "./features/features_dG4OHH.txt");
    get_dG4features(features_dG4HHO, parameters4_HHO, rij, vecrij, 1, 1, 0, "./features/features_dG4HHO.txt");
    get_dG4features(features_dG4HOO, parameters4_HOO, rij, vecrij, 1, 0, 0, "./features/features_dG4HOO.txt");

    // get dWG4 features
    get_dWG4features(features_dG4OHW,parameters4_OOH,rij,vecrij,nw_rij,vec_nw_rij,0,1,"./features/features_dG4OHW.txt");
    get_dWG4features(features_dG4HOW,parameters4_HOO,rij,vecrij,nw_rij,vec_nw_rij,1,0,"./features/features_dG4HOW.txt");

    return 0;

}
