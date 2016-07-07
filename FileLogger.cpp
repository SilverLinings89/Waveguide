

FileLogger::FileLogger() {

}

void FileLogger::start(){
	t.start();
}

void FileLogger::stop() {
	t.stop();
	if (solver) {
		file << fld.solver  << "\t";
	}
	if (preconditioner) {
			file << fld.preconditioner << "\t";
		}
	if (walltime) {
				file << t.wall_time() << "\t";
			}
	if (cputime) {
			file << t() << "\t";
		}
	if (XLength) {
			file << fld.XLength << "\t";
		}
	if (YLength) {
			file << fld.YLength << "\t";
		}
	if (ZLength) {
			file << fld.ZLength << "\t";
		}
	if (ParamSteps) {
			file << fld.ParamSteps << "\t";
		}
	if (Dofs) {
			file << fld.Dofs << "\t";
		}
	if (Precondition_BlockSize) {
			file << fld.Precondition_BlockSize << "\t";
		}
	if (PML_in) {
			file << fld.PML_in << "\t";
		}
	if (PML_out) {
			file << fld.PML_out << "\t";
		}
	if (PML_mantle) {
			file << fld.PML_mantle << "\t";
		}
	if (Solver_Precision) {
			file << fld.Solver_Precision << "\t";
		}
	if (Precondition_weight) {
			file << fld.Precondition_weight << "\t";
		}
	file << std::endl;
	file.close();
}

FileLogger::FileLogger( std::string in_filename,const FileLoggerData& logger){
	fld = logger;
	char tab2[1024];
	strncpy(tab2, in_filename.c_str(), sizeof(tab2));
	tab2[sizeof(tab2) - 1] = 0;
	bool exists = file_exists(in_filename);
	file.open(tab2, std::ios::app);
	if( ! exists ){

		if (solver) {
				file << "Solver"  << "\t";
			}
			if (preconditioner) {
					file << "Preconditioner" << "\t";
				}
			if (walltime) {
						file << "Walltime" << "\t";
					}
			if (cputime) {
					file << "CPU-time" << "\t";
				}
			if (XLength) {
					file << "X-Length" << "\t";
				}
			if (YLength) {
					file << "Y-Length" << "\t";
				}
			if (ZLength) {
					file << "Z-Length" << "\t";
				}
			if (ParamSteps) {
					file << "Steps" << "\t";
				}
			if (Dofs) {
					file << "Dofs" << "\t";
				}
			if (Precondition_BlockSize) {
					file << "PBlockSize" << "\t";
				}
			if (PML_in) {
					file << "PML in" << "\t";
				}
			if (PML_out) {
					file << "PML out" << "\t";
				}
			if (PML_mantle) {
					file << "PML mantle" << "\t";
				}
			if (Solver_Precision) {
					file << "Precision" << "\t";
				}
			if (Precondition_weight) {
					file << "P weight" << "\t";
			}

			file <<std::endl << "----------------------------------------" << std::endl;
	}
	solver = preconditioner = XLength = YLength = ZLength = ParamSteps = Dofs = Precondition_BlockSize = PML_in = PML_out = PML_mantle = Solver_Precision = Precondition_weight = walltime = cputime = false;
	walltime = true;
}

FileLogger::~FileLogger() {
	file.close();
}
