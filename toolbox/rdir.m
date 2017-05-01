function [varargout] = rdir(rootdir,varargin)
% RDIR - Recursive directory listing
% 
%  D = rdir(ROOT)
%  D = rdir(ROOT, TEST)
%  D = rdir(ROOT, TEST, RMPATH)
%  D = rdir(ROOT, TEST, 1)
%  D = rdir(ROOT, '', ...)
%  [D, P] = rdir(...)
%  rdir(...)
%
%
% *Inputs*
%
% * ROOT
%
% rdir(ROOT) lists the specified files.
% ROOT can be a pathname, filename, or can include both. One can use
% absolute and relative pathnames and wildcards (*). Wildcard can be placed
% anywhere and used many times like 'path*\*.m'
%
% One can also use a double wildcard (**) to match multiple directory
% levels. For example ROOT = 'path\**\*.m' will match all ".m" files in
% "path" and all subdirectories of "path".
% 
% NOTE : ".svn" directories created by SubVersion (SVN) are excluded from
% the recursive listing.
%
% * TEST
%
% Optional test that can be performed on the returned files. 
%
% TEST is a string indicating expression to be evaluated on selected field
% of rdir output.
% All fields (ie name, date, bytes, isdir and datenum) can be used.
%
% Tests are strings similar to what one would use in a "if" statement e.g.
%  'bytes>1024 & datenum>now-7' 
% 
% One can also use function like "regexp" or "strfind" with string fields
% like "name" and "date" e.g 'regexp(name, 'expr')'. In that case, tests
% that return a non empty value are considered as true.
%
% regexp(name, '(\.m$)|(\.mdl$)')
%
% Test can also be a function handle as used in arrayfun/cellfun, e.g.
% @(f)f.bytes>1024
%
% * RMPATH
%
% Optional path to remove from beginning of "name" field in returned
% output. Specified path must be common to all items found.
%
% If RMPATH = 1 or true, path to remove is part of ROOT before the first
% wildcard. 
%
%
% *Outputs*
%
% * D
%
% D is a structure with the same fields as Matlab DIR output. 
%
% The "name" field includes the relative path as well as the name to the
% file that was found. Path can be shorten or ommited when using 3rd
% argument RMPATH. 
%
% * P
%
% Common path or RMPATH (if specified) for the file list returned in D.
%
% * Screen output
%
% If not output variable is specified then the output is sent to the
% screen. 
%
%
% *Versions*
%
% * 1.0 - 2009, Gus Brown
% * 2.0 - 26/05/2011 Thomas Vanaret
%         No longer exclude all directories from a simple search (no *); 
%         Fixing bug on returned path; 
%         Exclude ".svn" directories; 
%         Extended test possibilies; 
%         Subfunctions created; 
% * 2.1 - 14/07/2011 Thomas Vanaret
%         New argument allowing to remove common path from name; 
%         Comments review; 
% * 2.2 - 20/12/2011 Thomas Vanaret
%         Fixing bug on display with 0b files;
%         Specific display when no file match filter;
% * 2.3 - 19/01/2014 Thomas Vanaret
%         Adding improvements suggested by X. Mo :
%         - function handle as TEST input
%         - code optimisation (avoiding loop)
%         Fixing possible bug when using a wildcard at the beginning;
%         Common path as 2nd optionnal output;
%
% 
% *Examples*
%
%   D = rdir('*.m');
%     for ii=1:length(D), disp(D(ii).name); end;
%
%   % to find all files in the current directory and sub directories
%   D = rdir('**\*')
%
%   % If no output is specified then the files are sent to 
%   % the screen.
%   rdir('c:\program files\windows *\*.exe');
%   rdir('c:\program files\windows *\**\*.dll');
%
%   % Using the test function to find files modified today
%   rdir('c:\win*\*','datenum>floor(now)');
%   % Using the test function to find files of a certain size
%   rdir('c:\program files\win*\*.exe','bytes>1024 & bytes<1048576');
%   % Using the test function to find files modified in 2011
%   rdir('c:\win*\*','strfind(date, ''2011'')');
%
%   % Using the 3rd input to shorten output name
%   rdir([matlabroot, '\*.txt'], '', 'C:\Program Files\')
%   % Using the 3rd input to shorten output name
%   rdir([matlabroot, '\*.txt'], '', 1)
%
%
% See also DIR
%


%--------------------------------------------------------------------------
%% Input validation

% use the current directory if nothing is specified
if ~exist('rootdir','var'),
  rootdir = '*';
end

prepath = '';       % the path before the wild card
wildpath = '';      % the path wild card
postpath = rootdir; % the path after the wild card
I = find(rootdir==filesep,1,'last');

% Directory separator for current platform
if filesep == '\'
  % On PC, filesep is '\'
  anti_filesep = '/';
else
  % On UNIX system, filesep is '/'
  anti_filesep = '\';
end

if isempty(I) && ~isempty(strfind(rootdir, anti_filesep))
  error([mfilename, ':FileSep'],...
    'Use correct directory separator "%s".', filesep)
end


%--------------------------------------------------------------------------
%% Split rootdir
% split the file path around the wild card specifiers

if ~isempty(I),
  prepath = rootdir(1:I);
  postpath = rootdir(I+1:end);
  I = find(prepath=='*',1,'first');
  if ~isempty(I),
    postpath = [prepath(I:end) postpath];
    prepath = prepath(1:I-1);
    I = find(prepath==filesep,1,'last');
    if ~isempty(I),
      wildpath = prepath(I+1:end);
      prepath = prepath(1:I);
    end;
    I = find(postpath==filesep,1,'first');
    if ~isempty(I),
      wildpath = [wildpath postpath(1:I-1)];
      postpath = postpath(I:end);
    end;
  end;
end;

% disp([' "' prepath '" ~ "' wildpath '" ~ "' postpath '" ']);

%--------------------------------------------------------------------------
%% Recursive listing
% Search for matching files until all wildcards have been considered.

if isempty(wildpath)
  % If no directory wildcards then just get files and directories list
  
  D = dir([prepath postpath]);

  % Exclude ".", ".." and ".svn" directories from the list
  excl = isdotdir(D) | issvndir(D);
  D(excl) = [];

  if isdir([prepath postpath]);
    fullpath = [prepath postpath];
  else
    fullpath = prepath;
  end
  
  % Place directories on the top of the list
  is_dir = [D.isdir]';
  D = [D(is_dir); D(~is_dir)];
  
  % Add path before name
  for ii = 1:length(D)
    D(ii).name = fullfile(fullpath, D(ii).name);
  end

  % disp(sprintf('Scanning "%s"   %g files found',[prepath postpath],length(D)));
  
elseif strcmp(wildpath,'**')
  % A double wildcards directory means recurs down into sub directories

  % first look for files in the current directory (remove extra filesep)
  D = rdir([prepath postpath(2:end)]);

  % then look for sub directories
  D_sd = dir([prepath '*']);

  % Exclude ".", "..", ".svn" directories and files from the list
  excl = isdotdir(D_sd) | issvndir(D_sd) | ~([D_sd.isdir]');
  D_sd(excl) = [];

  % Process each sub directory found
  % Performance tweak: avoid growing array within loop (X. Mo)
  c_D = arrayfun(@(x) rdir([prepath x.name filesep wildpath postpath]),...
    D_sd, 'UniformOutput', false);
  
  D = [D; cell2mat( c_D ) ];
  
else
  % Process directory wild card looking for sub directories that match
  
  D_sd = dir([prepath wildpath]);

  % Exclude ".", "..", ".svn" directories and files from the list
  excl = isdotdir(D_sd) | issvndir(D_sd) | ~([D_sd.isdir]');
  D_sd(excl) = [];
    
  if ~isdir(prepath) || ( numel(D_sd)==1 && strcmp(D_sd.name, prepath))
    % Fix case like rdir('path*\...') where prepath is not a full directoty
    % name OR case were prepath match a unique directory.
    % Previous "dir" return then the matching directory name(s).
    % prepath is cleaned to use them.
    %
    % In else case, prepath is a valid path which must be kept.
    prepath = '';
  end
  
  % Process each directory found  
  Dt = dir('');

  c_D = arrayfun(@(x) rdir([prepath x.name postpath]),...
    D_sd, 'UniformOutput', false);

  D = [Dt; cell2mat( c_D ) ];
  
end


%--------------------------------------------------------------------------
%% Apply filter
% If specified, apply the filter to refine the search.

nb_before_filt = length(D);
warning_msg = '';

if (nargin>=2 && ~isempty(varargin{1})),
  try
    if isa(varargin{1}, 'function_handle')
        test_tf = arrayfun(varargin{1}, D);
    else
        test_tf = evaluate(D, varargin{1});
    end
    
    D = D(test_tf);
    
  catch
    if isa(varargin{1}, 'function_handle')
      test_expr = func2str(varargin{1});
    else
      test_expr = varargin{1};
    end
    
    warning_msg = sprintf('Invalid TEST "%s" : %s', test_expr, lasterr);
  end
end


%--------------------------------------------------------------------------
%% Remove path
% If specified, remove given or common path from each returned path.

common_path = '';
if (nargin>=3 && ~isempty(varargin{2})),

  arg2 = varargin{2};
  if ischar(arg2)
    common_path = arg2;    
  elseif (isnumeric(arg2) || islogical(arg2)) && arg2
    common_path = prepath;    
  end
  
  rm_path = regexptranslate('escape', common_path);

  % Check that path is common to all 
  start = regexp({D.name}', ['^', rm_path]);
  
  % Convert to a logical.
  is_common = not( cellfun(@isempty, start) );

  if all(is_common)
    for k = 1:length(D)
      D(k).name = regexprep(D(k).name, ['^', rm_path], '');
    end
    
  else
    common_path = '';
  end
  
  % 19/07/2012 : ajouter common_path en sortie optionnelle
  
end


%--------------------------------------------------------------------------
%% Display listing if no output variables are specified
% Screen display.

nout = nargout;

if nout == 0
  if isempty(D)
    if nb_before_filt == 0
      fprintf('%s not found.\n', rootdir)
    else
      fprintf('No item matching filter.\n')
    end
  else
    
    if ~isempty(common_path)
     fprintf('All in : %s\n', common_path) 
    end
    
    pp = {'' 'k' 'M' 'G' 'T'};
    for ii = 1:length(D)
      if D(ii).isdir
        % Directory item : display name
        disp(sprintf(' %29s %-64s','',D(ii).name));
      else
        % File item : display size, modification date and name
        sz = D(ii).bytes;
        if sz > 0
          ss = min(4,floor(log2(sz)/10));
        else
          ss = 0;
        end
        disp(sprintf('%4.0f %1sb  %20s  %-64s ',...
          sz/1024^ss, pp{ss+1}, datestr(D(ii).datenum, 0), D(ii).name));
      end
    end
  end
elseif nout == 1
  % send list out
  varargout{1} = D;
else
  % send list and common path out
  varargout{1} = D;
  varargout{2} = common_path;
end;

if ~isempty(warning_msg)
  warning([mfilename, ':InvalidTest'],...
    warning_msg); % ap aff
end

%---------------------------- end of main function ------------------------
 

%% ------------------------------------------------------------------------
function tf = issvndir(d)
% True for ".svn" directories.
% d is a structure returned by "dir"
%

is_dir = [d.isdir]';

is_svn = strcmp({d.name}, '.svn')';
%is_svn = false; % uncomment to disable ".svn" filtering 

tf = (is_dir & is_svn);

%---------------------------- end of subfunction --------------------------

%% ------------------------------------------------------------------------
function tf = isdotdir(d)
% True for "." and ".." directories.
% d is a structure returned by "dir"
%

is_dir = [d.isdir]';

is_dot = strcmp({d.name}, '.')';
is_dotdot = strcmp({d.name}, '..')';

tf = (is_dir & (is_dot | is_dotdot) );

%---------------------------- end of subfunction --------------------------

%% ------------------------------------------------------------------------
function tf = evaluate(d, expr)
% True for item where evaluated expression is correct or return a non empty
% cell.
% d is a structure returned by "dir"
%

% Get fields that can be used
name = {d.name}'; %#ok<NASGU>
date = {d.date}'; %#ok<NASGU>
datenum = [d.datenum]'; %#ok<NASGU>
bytes = [d.bytes]'; %#ok<NASGU>
isdir = [d.isdir]'; %#ok<NASGU>

tf = eval(expr); % low risk since done in a dedicated subfunction.

% Convert cell outputs returned by "strfind" or "regexp" filters to a
% logical.
if iscell(tf)
  tf = not( cellfun(@isempty, tf) );
end

%---------------------------- end of subfunction --------------------------

%---------------------------- END OF FUNCTION -----------------------------
