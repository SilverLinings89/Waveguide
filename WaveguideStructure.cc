class WaveguideStructure {
	public:
		Sector[] case_sectors;
		double epsilon_K, epsilon_M;
		int sectors;
		double sector_z_length;
		double z_min, z_max;
		Tensor<2,3, double> TransformationTensor (double in_x, double in_y, double in_z);
		
	
	
}

class Sector {
	public:
		bool left;
		bool right;
		bool boundary;
		double r_0, r_1, v_0, v_1, m_0, m_1;
		double z_0, z_1;
		Tensor<2,3, double> TransformationTensorInternal (double in_x, double in_y, double in_z);
	
}

Tensor<2,3, double> WaveguideStructure::TransformationTensor (double in_x, double in_y, double in_z) {
	int idx = (in_z - z_min)/sector_z_length;
	return case_sectors[idx].TransformationTensorInternal(in_x, in_y, (in_z - z_min -idx*sector_z_length)/sector_z_length ); 
}

Tensor<2,3, double> Sector::TransformationTensorInternal (double in_x, double in_y, double z) {
	double temp = 1 / (r_0 - 3*r_0*z*z + 2*r_0*z*z*z + 3*r_1*z*z - 2*r_1*z*z*z);
	double zz = z*z;
	double zzz = zz*z;
	double zzzz = zz*zz;
	double zzzzz = zzz*zz;
	double zzzzzz = zzz*zzz;
	Tensor<2,3,double> u;
	u[0][0] = temp;
	u[0][1] = 0.0;
	u[0][2] = 0.0;
	u[1][0] = 0.0;
	u[1][1] = temp;
	u[1][2] = 0.0;
	u[2][0] = -(6*x*z*(r_0-r_1)*(z - 1))/(r_0 - 3*r_0*zz + 2*r_0*zzz + 3*r_1*zz - 2*r_1*zzz);
	u[2][1] = 0.0;
	u[2][2] = 1.0;
	 
	Tensor<2,3,double> ginv;
	for(int i = 0; i<3; i++) {
		for(int j = 0; j<3; j++) {
			for(int k = 0; k< 3; k++) ginv[i][j] += u[i][k] * u[j][k];
		}
	}
	 
	double det = ginv[0][0]*( ginv[2][2]*ginv[1][1] - ginv[2][1]*ginv[1][2]) - ginv[1][0]*(ginv[2][2]*ginv[0][1] - ginv[2][1]*ginv[0][2]) + ginv[2][0]*(ginv[1][2]*ginv[0][1] - ginv[1][1]*ginv[0][2];
	
	Tensor<2,3,double> g;
	g[0][0] = (ginv[2][2] * ginv[1][1] - ginv[2][1]*ginv[1][2]);
	g[0][1] = - (ginv[2][2] * ginv[0][1] - ginv[2][1]*ginv[0][2]);
	g[0][2] = (ginv[1][2] * ginv[0][1] - ginv[1][1]*ginv[0][2]);
	g[1][0] = - (ginv[2][2] * ginv[1][0] - ginv[2][0]*ginv[1][2]);
	g[1][1] = (ginv[2][2] * ginv[0][0] - ginv[2][0]*ginv[0][2]);
	g[1][2] = - (ginv[1][2] * ginv[0][0] - ginv[1][0]*ginv[0][2]);
	g[2][0] = (ginv[2][1] * ginv[1][0] - ginv[2][0]*ginv[1][1]);
	g[2][1] =  -(ginv[2][1] * ginv[0][0] - ginv[2][0]*ginv[0][1]);
	g[2][2] = (ginv[1][1] * ginv[0][0] - ginv[1][0]*ginv[0][1]);
	
	
}

