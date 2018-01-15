function openfigure(figno,vargin,screentype)
%% CALL: openfigure(figno,vargin,screentype);
% INPUT:
%    figno ... INT; Figure number.
%    varargin ... CELL; Variable list for plot.
%    screentype ... INT; 0: Plots in different figures,
%                        1: All plots in one figure.
% OUTPUT:
%    Opens/activates windows or draws into windows.
% DESCRIPTION:
% Organizes the figures. Nf(<=8) Figures are opened by
%    openfigure(Nf,'init');
% They are placed side-by-side in one or two rows. This total number of
% figures is memorized for all further calls.
%    openfigure(Nf,'init',screentype);
% sets onescreen=screentype, screentype=0 the default.
% The meaning of onescreen is:
%    onescreen==0: Figure figno is opened/activates individually
%    onescreen==1: Figure figno is one of Nf subplots of Figure 1 
% Call
%    openfigure(figno)
% to activate window figno or use
%    openfigure(figno,{'clf','hold on'})
% to apply further initialisations of window figno.

% Developed at IANM 2, Univ. Karlsruhe. Matlab 7.2.0.294 (R2006a).
% Last modified: 27.02.2013, W. Doerfler.

%% Parameter
persistent Nf onescreen;% Memorize between calls
sh = 10;      % Distance between figures
bsh = 50;     % Bottom edge distance
rsh = 0;      % Right edge distance
msh = 75;     % Menu bar adjustment

%% Open
if nargin==0
   ng = 0;
   Nf = 1;
   figno = 1;
   vargin = {'init'};
   onescreen = 0;% Default
elseif nargin==1
   ng = 1;
   vargin = {};
elseif nargin==2
   ng = 2;
   if strcmp(vargin,'init'), onescreen = 0; end% Default
elseif nargin==3
   ng = 2;
   if ~strcmp(vargin,'init')
      error('*** 3 args with option init only ***'); 
   end
   onescreen = screentype;
end
if ng==0 || ng==2
   if strcmp(vargin,'init')
      Nf = min(figno,8);% Number of figures reserved (retrict to 8 for safety)
      % Initialize graphics
      scrsz = get(0,'ScreenSize'); % Get screensize
      scrsz = 0.9*scrsz;% Shrink
      if onescreen==1
         hF = figure(1); clf; % Open figure
         xw = scrsz(3)/2; yw = 3*scrsz(4)/4; % Calculate windowsize
         x = xw/2; y = scrsz(4)-yw-msh-sh; % Calculate figure coordinates
         set(hF,'Position',[x y xw yw]); % Set figure coordinates
         for k=1:Nf, subplot(ceil(Nf/2),2,k,'replace'); end % Clear 1-Nf of Nf
         return
      else
         if Nf>2
            yw = (scrsz(4)-bsh-2*(sh+msh))/2; % Calculate windowsize
            xw = (scrsz(3)-rsh-(ceil(Nf/2)+1)*sh)/ceil(Nf/2); % Calculate windowsize
            coord = zeros(Nf,2); 
            for i=1:Nf
               coord(i,1:2) = [sh+floor((i-1)/2)*(sh+xw), ...
                  scrsz(4)-yw-sh-msh-(mod(i+1,2))*(sh+yw+msh)]; % Calculate figure coordinates
            end
         else
            yw = scrsz(4)-bsh-sh-msh; xw=(scrsz(3)-rsh-2*sh)/Nf; % Calculate windowsize
            coord = [sh, scrsz(4)-yw-sh-msh;2*sh+xw, scrsz(4)-yw-sh-msh]; % Calculate figure coordinates
         end
         for i=1:Nf
            set(figure(i),'Position',[coord(i,1:2) xw yw]); clf; % Open figure        
         end
         return
      end
   end
end

%% Draw
%try % Use a single window figno if figno>Nf
if ~isempty(Nf)
   if figno>Nf
      figure(figno);
      for i = 1:length(vargin)
         eval(vargin{i});
      end
      return
   end
%catch % Do the same if Nf is undefined %% Achtung: Springt nicht in catch!
else
   figno = min(figno,16);% Safety
   figure(figno);
   for i=1:length(vargin)
      eval(vargin{i});
   end
   return
end
if onescreen==0
   figure(figno);
   for i=1:length(vargin)
      eval(vargin{i});
   end
else
   figure(1); subplot(ceil(Nf/2),2,figno,'replace');
   for i=1:length(vargin)
      if ~strcmp(vargin{i},'clf'), eval(vargin{i}); end
   end
end

return
