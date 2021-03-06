%% Import data from spreadsheet
% Script for importing data from the following spreadsheet:
%
%    Workbook: /Users/youssufsaleh/Dropbox (Neurological Conditions)/CHAPAS data (1)/questionnaire/Questionnaire_full.xlsx
%    Worksheet: Sheet1
%
% Auto-generated by MATLAB on 25-Apr-2019 10:52:08

%% Setup the Import Options
opts = spreadsheetImportOptions("NumVariables", 135);

% Specify sheet and range
opts.Sheet = "Sheet1";
opts.DataRange = "A2:EE41";

% Specify column names and types
opts.VariableNames = ["participantD", "SHAPS1", "SHAPS2", "SHAPS3", "SHAPS4", "SHAPS5", "SHAPS6", "SHAPS7", "SHAPS8", "SHAPS9", "SHAPS10", "SHAPS11", "SHAPS12", "SHAPS13", "SHAPS14", "SHAPS_TOTAL", "AMI1", "AMI2", "AMI3", "AMI4", "AMI5", "AMI6", "AMI7", "AMI8", "AMI9", "AMI10", "AMI11", "AMI12", "AMI13", "AMI14", "AMI15", "AMI16", "AMI17", "AMI18", "AMI_TOTAL", "BNSS1", "BNSS2", "BNSS3", "BNSS_ANHEDONIA", "BNSS4", "BNSS_DISTRESS", "BNSS5", "BNSS6", "BNSS_ASOCIALITY", "BNSS7", "BNSS8", "BNSS_AVOLITION", "BNSS9", "BNSS10", "BNSS11", "BNSS_BLUNTED", "BNSS12", "BNSS13", "BNSS_ALOGIA", "BNSS_TOTAL", "CALGARY_DEPRESSION", "CALGARY_HOPELESS", "CALGARY_SELFDEPRECIATE", "CALGARY_GUILTY_IDEAS", "CALGARY_PATHOLOGICAL_GUILT", "CALGARY_MORNING_DEPRESS", "CALGARY_EARLY_WAKE", "CALGARY_SUICIDE", "CALGARY_OBSERVED_DEPRESS", "CALGARY_TOTAL", "PANSS_P1", "PANSS_P2", "PANSS_P3", "PANSS_P4", "PANSS_P5", "PANSS_P6", "PANSS_P7", "PANSS_POS_TOTAL", "PANSS_N1", "PANSS_N2", "PANSS_N3", "PANSS_N4", "PANSS_N5", "PANSS_N6", "PANSS_N7", "PANS_NEG_TOTAL", "PANSS_G1", "PANSS_G2", "PANSS_G3", "PANSS_G4", "PANSS_G5", "PANSS_G6", "PANSS_G7", "PANSS_G8", "PANSS_G9", "PANSS_G10", "PANSS_G11", "PANSS_G12", "PANSS_G13", "PANSS_G14", "PANSS_G15", "PANSS_G16", "PANSS_GEN_TOTAL", "PANSS_GRAND_TOTAL", "BACS_VM_Z", "BACS_VM_T", "BACS_DS_Z", "BACS_DS_T", "BACS_TMT_Z", "BACS_TMT_T", "BACS_VF_Z", "BACS_VF_T", "BACS_SC_Z", "BACS_SC_T", "BACS_ToL_Z", "BACS_ToL_T", "BACS_TOTAL_Z", "BACS_TOTAL_T", "ACE_ATTENTION", "ACE_MEMORY", "ACE_FLUENCY", "ACE_LANGUAGE", "ACE_VISUOSPATIAL", "ACE_TOTAL", "PSP_TOTAL", "DoB", "Gender", "DoFEP", "DoTesting", "Dose", "Clz_level", "Nor_Clz", "OtherM1", "OtherM2", "Notes", "BMI", "T2DM", "GlucoseDysregulation", "Age", "illnessduration"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "datetime", "double", "double", "datetime", "double", "double", "double", "string", "string", "string", "double", "double", "double", "double", "double"];
opts = setvaropts(opts, [128, 129, 130], "WhitespaceRule", "preserve");
opts = setvaropts(opts, [128, 129, 130], "EmptyFieldRule", "auto");

% Import the data
Questionnaires = readtable("/Users/youssufsaleh/Dropbox (Neurological Conditions)/CHAPAS data (1)/questionnaire/Questionnaire_full.xlsx", opts, "UseExcel", false);


%% Clear temporary variables
clear opts

%% This next section plots the correlation matrix between questionnaires.
% It is worth defining some of these questionnaires as I am unfamiliar with
% quite a few off them: 

% 1. BACS: Brief assessment of cognition in Schizophrenia. 
% 2. PANS: Positive and negative syndrome scale.
% 3. BNSS: Brief negative symptoms scale. 
% 4. AMI: Apathy Motivation index. THis has three components which need to
% be calculated. 

%create subcomponents of the AMI which are: Behavioural, Emotional and Social.
% I will call these: AMI.B,AMI.E, and AMI.S. 

% Each subcomponent contains the sum of the following questionnaire components: 
% AMI.B  = 5,9,10,11,12,15 
% AMI.E  = 1,6,7,13,16,18
% AMI.S = 2,3,4,8,14,17
% These are then divided by 6. The cut offs can be found in Yuens's PLOS
% one paper "Distinct Subtypes of Apathy Revealed by the Apathy Motivation
% Index"

%Calculate subcomponents from my Questionnaire. First for Behavioural
%component
AMI.B = (Questionnaires.AMI5 + Questionnaires.AMI9 + ...
    Questionnaires.AMI10 + Questionnaires.AMI11 +    ...
    Questionnaires.AMI12 + Questionnaires.AMI15)./6;
%Emotional
AMI.E = (Questionnaires.AMI5 + Questionnaires.AMI9 + ...
    Questionnaires.AMI10 + Questionnaires.AMI11 +    ...
    Questionnaires.AMI12 + Questionnaires.AMI15)./6;
% Social
AMI.S = (Questionnaires.AMI2 + Questionnaires.AMI3 + ...
    Questionnaires.AMI4 + Questionnaires.AMI8 +    ...
    Questionnaires.AMI14 + Questionnaires.AMI17)./6;
% Also calculate the total in the way it was published, ie. divide by 18. 
AMI.Total = Questionnaires.AMI_TOTAL./18;

% now add them to the big questionnaire. nb doing it this way adds them ...
% on to the end of the questinonaire. 
Questionnaires.AMI_B = AMI.B;
Questionnaires.AMI_E = AMI.E;
Questionnaires.AMI_S = AMI.S;
Questionnaires.AMI_T = AMI.Total;

% the SHAPS, which is the anxiety scale was incorrectly scored. It is
% usually negatively scored so I will correct this as well. 
SHAPS_Corrected = (Questionnaires.SHAPS_TOTAL).*-1;
Questionnaires.SHAPS_Corrected = SHAPS_Corrected;


% I was also informed by emilio that the best representative score for
% negative symptoms is the sum of the first three questions of the BNSS. 

Questionnaires.BNSS_neg = Questionnaires.BNSS1 + ...
    Questionnaires.BNSS2 + ...
    Questionnaires.BNSS3;
% plot correlation matrix. for this function to work make sure you first
% remove 'matlib' from the path. I will embed this in the script to make
% life easier and add it back at the end. 
% remove matlib 
rmpath('/Users/youssufsaleh/Documents/Master folder/Apples v2/matlib')

% produce correlation plots. 
[R,PValue,H] = corrplot(Questionnaires(:,{'ACE_TOTAL', 'BACS_TOTAL_T',  ...
'AMI_T','AMI_B','AMI_E','AMI_S','BNSS_ANHEDONIA','BNSS_AVOLITION',      ...
    'CALGARY_TOTAL','PANS_NEG_TOTAL','BNSS_TOTAL','SHAPS_Corrected',    ...
    'Clz_level','Dose','BMI','Age'}),                                   ...
    'type','Pearson','testR','on','rows','pairwise');

% save image as png
saveas(gcf,'CHAPAS_Correlations.png')
% add matlib back as I commonly use its functions in other parts of the
% script. 
addpath('/Users/youssufsaleh/Documents/Master folder/Apples v2/matlib')


% Next is to just calculate all of the demographics for this group.
means = nanmean([Questionnaires.Age Questionnaires.ACE_TOTAL ...
    Questionnaires.BACS_TOTAL_T Questionnaires.CALGARY_TOTAL ...
    Questionnaires.PANS_NEG_TOTAL Questionnaires.BNSS_TOTAL  ...
    Questionnaires.AMI_T Questionnaires.BMI]);

% Standard deviations
stds = nanstd([Questionnaires.Age Questionnaires.ACE_TOTAL ...
    Questionnaires.BACS_TOTAL_T Questionnaires.CALGARY_TOTAL ...
    Questionnaires.PANS_NEG_TOTAL Questionnaires.BNSS_TOTAL  ...
    Questionnaires.AMI_T Questionnaires.BMI]);

% then split them up by apathy. Cut offs that can be used: 
% Total AMI: Moderate >= 1.91 Severe > = 2.37. Behavioural score >= 2.34 if
% moderate and >=3.09 if severe. There are 15 patients who are moderately
% apathetic for both total and behavioural. Only 2-4 score significantly
% for the severe apathy for either. I think it is reasonable to use
% moderate in that case. You can manipulate the line below to check. 
size(find(Questionnaires.AMI_T>1.90));

% lets get them all sorted and compared by group now. 

subj = size(Questionnaires,1); %How many subjects?
apVec=[]';
for i = 1:subj
    if Questionnaires.AMI_T(i) > 1.90
        apVec(i)=1;
    else apVec(i)=0;
    end
end
    
apVec = apVec';

% what I want now is to create a demographics table. 
% first add the grouping variable into the table 
Questionnaires.groupAlloc = apVec;
Qs = Questionnaires(:,{'Age','ACE_TOTAL','BACS_TOTAL_T', ...
    'CALGARY_TOTAL', 'PANS_NEG_TOTAL','BNSS_TOTAL',      ...
    'AMI_T','BMI','groupAlloc'});

% Demographics table
clear m s M S t
m=varfun(@nanmean,Qs(:,:),'GroupingVariable',{'groupAlloc'});
s=varfun(@nanstd,Qs(:,:),'GroupingVariable',{'groupAlloc'});

[h p] = ttest2(table2array(Qs(Qs.groupAlloc==1,1:8)), ...
  table2array(Qs(Qs.groupAlloc==0,1:8)));

M = table2array(m(:,3:end));
S = table2array(s(:,3:end));

t = array2table(horzcat(M(1,:)',S(1,:)',M(2,:)',S(2,:)',p'));
t.Properties.VariableNames = {'lessApathetic_mean','lessApathetic_SD',...
    'moreApathetic_mean','moreApathetic_SD','Pvalue'};

t.Properties.RowNames = {'Age','ACE','BACS','CALGARY','PANSN', ...
    'BNSST','AMI','BMI'};
 
writetable(t,...
   'CHAPASDemographics.csv',...
    'WriteRowNames',true);
% create table for chisquared test. 
Chi = array2table(horzcat(Questionnaires.Gender, Qs.groupAlloc));
Chi.Properties.VariableNames = {'Gender','apathy'};

writetable(Chi,...
   'CHAPASGENDER.csv',...
    'WriteRowNames',true);


% good that is a good start. This script seems to get us most of the
% basic demographics. 



























% Next I want to process all the data and run the first glme using several
% models that vary apathy in its compnents, cut offs and continuous nature.
% If there is some more time we can then try and run the computational
% models. 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
