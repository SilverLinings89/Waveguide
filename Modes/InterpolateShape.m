function [output] = InterpolateShape(fname)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    numShapes = 36;
    files = dir('*.hdf5');
    output = zeros(36,796);
    for index=1:numShapes
        tempdata = hdf5read(files(index).name, '/pwb/trajectory/coordinates');
        count = size(tempdata(1,:),2);
        output(index, 1) = count;
        output(index,2) = tempdata(1,1);
        output(index,count +2) = tempdata(3,1);
        output(index,2*count +2) = 0;
        for i=2:count-1
            output(index,i+1)= tempdata(1,i);
            output(index,i+1+count)= tempdata(3,i);
            output(index,i+1+2*count)= ((tempdata(3,i+1)-tempdata(3,i))/(tempdata(1,i+1)-tempdata(1,i)) + (tempdata(3,i)-tempdata(3,i-1))/(tempdata(1,i)-tempdata(1,i-1)))/2.0;
        end
        output(index,count+1) = tempdata(1,count);
        output(index,2*count+1) = tempdata(3,count);
        output(index,3*count+1) = 0;
    end
    dlmwrite(fname, output);
%     fid = fopen(fname, 'w') ;
%     for i=1:36
%         for j =1:796
%             fprintf(fid, '%s\t', output(i,j)) ;
%         end
%         fprintf(fid, '\n');
%     end
%     fclose(fid) ;
end