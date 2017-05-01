%% RDIR Enhanced - Examples of use
%
% This script demonstrates how to use the different abilities of the
% enhanced |rdir| function.
%
% Examples are based on |matlabroot| directory content. Results may vary
% depending on your version of Matlab. 
%

%% Standard use
rdir([matlabroot, '\*.txt'])

%% Using double wildcard **
% List |".m"| files whose name contains |"tmpl"| in all subdirectories of
% |matlabroot|
rdir([matlabroot, '\**\*tmpl*.m'])

%% RDIR output
d = rdir([matlabroot, '\**\*tmpl*.m'])

%%
disp(d(1))


%% Using 3rd argument to shorten output names
% Remove |"C:\Program Files\"| in returned names
rdir([matlabroot, '\*.txt'], '', 'C:\Program Files\')

%%
% Remove |matlabroot| in returned names
rdir([matlabroot, '\*.txt'], '', true)

%%
% Optional 2nd |rdir| output indicates common path removed from each output
% name
[d, p] = rdir([matlabroot, '\*.txt'], '', true);

fprintf('Common path : \n%s\n\n', p)

disp( d(1) )

%% Using a filter with "regexp"
% List |".mat"| files, then select those whose name match regular expression
% |'data\d'| (ie |"data"| followed by a numeric digit)
rdir([matlabroot '\toolbox\**\*.mat'], 'regexp(name, ''data\d'')', true)

%% Using a function handle as filter

fun = @(d) ~isempty(regexp(d.name, 'data\d')) && (d.bytes < 10*1024)

rdir([matlabroot '\toolbox\**\*.mat'], fun, true)

%% Specific display - No item matching filter
% When some items match input path, but none match filter, a specific
% message is displayed.
rdir(matlabroot, 'strcmp(name, ''unknowtoolbox'')', 1)


%% Specific display - Wrong filter
% A warning is displayed after the non-filtered result list if entered
% filter is wrong.
rdir(matlabroot, 'wrong filter', 1)


% EOF