%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: /Users/youssufsaleh/Downloads/SVD-5.xlsx
%    Worksheet: Demographics_Final
%
% To extend the code for use with different selected data or a different
% spreadsheet, generate a function instead of a script.

% Auto-generated by MATLAB on 2019/03/04 15:23:05

%% Import the data
[~, ~, raw] = xlsread('/Users/youssufsaleh/Downloads/SVD-5.xlsx','Demographics_Final');
raw = raw(2:end,:);
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
stringVectors = string(raw(:,[1,16,17,18,19]));
stringVectors(ismissing(stringVectors)) = '';
raw = raw(:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15]);

%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
data = reshape([raw{:}],size(raw));

%% Create table
Qs_Final_raw = table;

%% Allocate imported array to column variable names
Qs_Final_raw.OSVID = stringVectors(:,1);
Qs_Final_raw.Age = data(:,1);
Qs_Final_raw.GenderM1 = data(:,2);
Qs_Final_raw.LARS_E = data(:,3);
Qs_Final_raw.LARS_AI = data(:,4);
Qs_Final_raw.LARS_SA = data(:,5);
Qs_Final_raw.LARS_TOTAL = data(:,6);
Qs_Final_raw.AES_TOTAL = data(:,7);
Qs_Final_raw.AES_Cognitive = data(:,8);
Qs_Final_raw.AES_Behavioural = data(:,9);
Qs_Final_raw.AES_Emotional = data(:,10);
Qs_Final_raw.Other = data(:,11);
Qs_Final_raw.BDI = data(:,12);
Qs_Final_raw.ACE_Total = data(:,13);
Qs_Final_raw.Excluded = data(:,14);
Qs_Final_raw.FullStructural = stringVectors(:,3);
Qs_Final_raw.FullDiffusion30 = stringVectors(:,4);
Qs_Final_raw.FullDiffusion60 = stringVectors(:,5);

%% Clear temporary variables
clearvars data raw stringVectors R;



%% Extract excluded subjects
FQs_Ex= Qs_Final_raw(Qs_Final_raw.Excluded==0,:);

% Having done this, what we want to look at is a single latent variable
% that looks at the combined effect of the LARS and AES. Here I will use
% the LARS - Action Initiation. 
[coeff,score,latent,tsquared,explained,mu]=pca(nanzscore ...
    ([FQs_Ex.AES_TOTAL,FQs_Ex.LARS_AI]),'NumComponents',1);
Composite = score;

% lets add this onto the parent table. nb. 'addvars' is a function that needs
% MATLAB version 2018a and beyond. 
FQs_Ex = addvars(FQs_Ex,Composite,'After','Other');
% now lets have a look at how this measures up against the rest of the
% variables

save('Questionnaires_final','FQs_Ex','Qs_Final_raw');



%% Correlation plots 

% Before doing this you should temporarily remove matlib from your path as
% it contains a function with the same name that does something different. 
rmpath     '/Users/youssufsaleh/Documents/Master folder/Apples v2/matlib'

[R,PValue] = corrplot(FQs_Ex(:,{'Age','LARS_TOTAL' 'LARS_E','LARS_AI','LARS_SA', ...
 'AES_TOTAL','AES_Cognitive','AES_Behavioural', ...
 'AES_Emotional','Other','Composite','BDI','ACE_Total'}), ...
 'type','Pearson','testR','on','rows','pairwise');




% What I want to do now is add on the accept variable from the analysis
% script to see how this correlates with my different questionnaires. 

Qs_Accept = addvars(FQs_Ex,accept,'After','Excluded');

[R,PValue] = corrplot(Qs_Accept(:,{'Age','LARS_TOTAL' 'LARS_E','LARS_AI','LARS_SA', ...
 'AES_TOTAL','AES_Cognitive','AES_Behavioural', ...
 'AES_Emotional','Other','Composite','BDI','ACE_Total','accept'}), ...
 'type','Pearson','testR','on','rows','pairwise');


% Add matlib back on as it is used regularly for the
% rest of the analysis. 
addpath     '/Users/youssufsaleh/Documents/Master folder/Apples v2/matlib'
savepath pwd























