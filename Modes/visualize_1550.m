T = readtable('mode_1550nm.dat');
M = table2array(T);
openfigure(6, 'init');
titles= ["Realteil: E_z","Realteil: E_x","Realteil: E_y","Imaginaerteil: E_z","Imaginaerteil: E_x","Imaginaerteil: E_y"];
for i=1:6
    dat = xyz2grid(M(:,2),M(:,3),M(:,i+3));
    openfigure(i);
    
    surf(dat,'EdgeColor','none','LineStyle','none','FaceLighting','phong');
    title(titles(i));
end
