%% Script for Youssuf Saleh's Small Vessel Disease paper and thesis.  
%% Intention is for this to be the 'final' script for Small vessel disease 
% This includes data pre-processing, visualisation of data and analysis as
% well as the produciton of key MRI regressors. 
%% Performing relevant analyses including computational modelling. 
% 1. Converting excel sheet into a matlab table and extracting on excluded
% data points. 
% 2. Plotting a correlations table of all the questionnaires. 
% 3. Using a median split to look at 




% first of all I want a simple visualisation of the entire group's
% behavioura over the  course of the task. 

% add shaded errorbar function to path and save. 
addpath  '/Users/youssufsaleh/Documents/Master folder/Apples v2/raacampbell-shadedErrorBar-0dc4de5/'
savepath pwd
% create function handle for function which controls colors
c = @cmu.colors;

close all
grp_rew_2D = squeeze(mean(freqmap,1))';
H1=shadedErrorBar(1:6,nanmean(grp_rew_2D),   ...
    nanstd(grp_rew_2D./sqrt(size(D.R,1))),'lineprops',...
    {'-.','color', c('royal purple')},...
    'patchSaturation',0.3);
hold on
grp_eff_2D = squeeze(mean(freqmap,2))';
H2=shadedErrorBar(1:6,nanmean(grp_eff_2D),   ...
    nanstd(grp_eff_2D./sqrt(size(D.R,1))),'lineprops',...
    {'-.','color',c('air force blue')},  ...
    'patchSaturation',0.3);
axis square
ax=gca;
set(ax,'fontWeight','bold','fontSize',18,'XTick',[1:1:5], ...

    'XTickLabel',{'1','3','6','9','12','15'})
xlabel('Reward (Apples)')
ylabel('Prop. Offers Accepted')
ylim([0 1.1]); xlim([0.5 6.5]);xticks([1 2 3 4 5 6]);
title('Reward incentivises motivated behaviour');
% 
%     'XTickLabel',{'10','24','38','52','66','80'})
% xlabel('Effort (%MVC)')
% ylabel('Prop. Offers Accepted')
% title('Effort deters motivated behaviour')
% ylim([0 1]); xlim([0.5 6.5]);xticks([1 2 3 4 5 6]);
% title('Group average performance');
hold off


[lgd, icons, plots, txt] = legend([H1.mainLine],{'Reward'});


% I now want to quality check every single subjects behaviour throughout
% the task. 
subjects = [1:83];
close all
% now for each individual. 
% create matrices compatible with the errorbar function in order to
% generate errorbars for each individual's plots. 
% first rearrange the choice map so that we can have each subjects choice
% data individually available to create error bars for each. 
permute_choicemap_rew = squeeze(mean(permute(choicemap,[1,2,4,3]),1));
permute_choicemap_eff = squeeze(mean(permute(choicemap,[1,2,4,3]),2));
dat_rew = permute(permute_choicemap_rew,[2,1,3]);
dat_eff = permute(permute_choicemap_eff,[2,1,3]);

% now generate a subplot for each individual with their performance. 
%create legend indices
close all
subj = size(D.R,1);

%exclude = [which ever subvject number you want to exclude]; You can use
%this as a handle to index into and look closer at those not behaving
%typically. 
for i = 1:subj
  subplot(10,9,i)
  hold on
  H3= shadedErrorBar(1:6,nanmean(dat_rew(:,:,i),1), ...
    nanstd(dat_rew(:,:,i)./sqrt(5)),'lineprops',...
    {'color', c('royal purple')},...
    'patchSaturation',0.3);
  hold on
  H4= shadedErrorBar(1:6,nanmean(dat_eff(:,:,i),1), ...
    nanstd(dat_eff(:,:,i)./sqrt(5)),'lineprops',...
    {'color', c('air force blue')},...
    'patchSaturation',0.3);
  hold off
  hold off
  axis square
  ylim([0 1.1]); xlim([0.5 6.5]);xticks([1 2 3 4 5 6]);
  title(['Subject ' num2str(subjects(i))],'FontSize',8);
  %ax=gca;
  %set(ax,'fontWeight','bold','fontSize',12,'XTick',[1:1:5], ...
    %xticksabel({'1','2','3','4','5','6'});
end

[lgd, icons, plots, txt] = legend([H3.mainLine H4.mainLine],...
  {'Reward','Effort'});

 [ax1,h1]=suplabel('Effort/Reward Level');
 [ax2,h2]=suplabel('Prop. Accepted','y');




close all

% Load Questionnaire data after exlu

% exclude RJ from SVD cohort. 

% create apathy vector (~debatable on exactly how to categorise patients)
dep14=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.BDI(i) > 13
            %nanstd(FQs_Ex.Composite)
        dep14(i)=1;
    else dep14(i)=0;
    end
end
    
dep14 = dep14';
% set up another cut off 
dep19=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.BDI(i) > 19
            %nanstd(FQs_Ex.Composite)
        dep19(i)=1;
    else dep19(i)=0;
    end
end
    dep19 = dep19';

apVec37 = apVec37';
% and another at the median. 
apVec29=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.AES_TOTAL(i) > 29
            %nanstd(FQs_Ex.Composite)
        apVec29(i)=1;
    else apVec29(i)=0;
    end
end
    
apVec29 = apVec29';

apMedian=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.AES_TOTAL(i) >= nanmedian(FQs_Ex.AES_TOTAL)
            %nanstd(FQs_Ex.Composite)
        apMedian(i)=1;
    else apMedian(i)=0;
    end
end
    
apMedian = apMedian';
%apVec(9)=1; %aesVec for this subject 40
depMedian=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.BDI(i) > nanmedian(FQs_Ex.BDI)
            %nanstd(FQs_Ex.Composite)
        depMedian(i)=1;
    else depMedian(i)=0;
    end
end
    
depMedian = depMedian';

apVec37=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.AES_TOTAL(i) > 37
            %nanstd(FQs_Ex.Composite)
        apVec37(i)=1;
    else apVec37(i)=0;
    end
end
    
apVec37 = apVec37';

apVec34=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.AES_TOTAL(i) > 34
            %nanstd(FQs_Ex.Composite)
        apVec34(i)=1;
    else apVec34(i)=0;
    end
end
    
apVec34 = apVec34';

apVec365=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.AES_TOTAL(i) > 36.5
            %nanstd(FQs_Ex.Composite)
        apVec365(i)=1;
    else apVec365(i)=0;
    end
end
    
apVec365 = apVec365';

apVec40 = apVec34';

apVec40=[]';
for i = 1:subj
    %if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    if FQs_Ex.AES_TOTAL(i) > 40
            %nanstd(FQs_Ex.Composite)
        apVec40(i)=1;
    else apVec40(i)=0;
    end
end
    
apVec40 = apVec40';
    
% reshape choice data & prepare decision time matrix
choices = nanmean(grpD.choicemap{1},4);
forces = nanmean(grpD.maxforce{1},4);%all subjects average choices with accidental squeezes removed (nan)
%times = nanmean(grpD.log_DT,4);
dt = D.endChoice - D.startChoice; % decision time info
dt = dt(:,37:end); %exclude practice session

% Get rid of first block for all relevant matrices
rawYes=D.Yestrial; rawYes(isnan(rawYes))=0;
D.Yestrial=D.Yestrial(:,37:end);
D.stake = D.stake(:,37:end);
D.effort = D.effort(:,37:end);
D.reward = D.reward(:,37:end);
D.maximumForce = D.maximumForce(:,37:end);
%D.vigour = D.vigour(:,37:end);

decisionTime=dt; % exclude accidental squeezes below from this matrix
if 0
%Check some MVC metrics
close all;
meanMVC = [D.MVC1 D.MVC2];
meanMVC = mean(meanMVC,2);
scatterRegress(FQs_Ex.AES_TOTAL,meanMVC);
xlabel('LARS - SELF')
ylabel('meanMVC (Newtons)')
title('No correlation between MVC and apathy score')
set(gca,'fontSize',16,'fontWeight','bold')
end

% Offers accepted
% First do Raw offers accepted, then proportional (as if accidental squeeze
% unfair to code this as a "reject")
close all
yesTrial = D.Yestrial; % manipulate new variable yesTrial

yesTrial(isnan(yesTrial))=0; % Code for 'No' = 0
 
for i=1:size(D.R,1)
    for t=1:180
        if dt(i,t)<0.4  % if squeezed accidentally
            yesTrial(i,t)=nan;
            decisionTime(i,t)=nan;
            
        end
    end
end

% and now each '1' is a an accept, each '0' is a reject and each nan is a
% mistaken squeeze
 % here, NaN means iether rejected or didn't have to squeeze, and 0 means accepted but didn't achieve required force
for i=1:size(D.R,1)
    grpD.accept(i) = length(find(yesTrial(i,:)==1));
    grpD.reject(i) = length(find(yesTrial(i,:)==0));
    grpD.mistake(i)= length(find(isnan(yesTrial(i,:))));
    grpD.trialsCorr(i)= length(find(~isnan(yesTrial(i,:)))); % how many trials after removing mistakes
    grpD.fail(i) = length(find(yesTrial(i,:)==1 & D.stake(i,:)==0));
    grpD.failCorr(i) = length(find(yesTrial(i,:)==1 & D.stake(i,:)==0))/grpD.trialsCorr(i);
    grpD.failHighEff(i) = length(find(yesTrial(i,:)==1 & D.stake(i,:)==0 & D.effort(i,:)>0.6))/grpD.trialsCorr(i); % this metric erroneous as divides by ALL trials
    grpD.failHighEff2(i) = length(find(yesTrial(i,:)==1 & D.stake(i,:)==0 & D.effort(i,:)>0.6));
end

accept=grpD.accept';

% And plot this
figure()
bar(1,mean(accept(apVec==0)),0.5);hold on;bar(2,mean(accept(apVec==1)),0.5);
errorbar([mean(accept(apVec==0)) mean(accept(apVec==1))],[std(accept(apVec==0))/sqrt(length(find(apVec==0))) ...
    std(accept(apVec==1))/sqrt(length(find(apVec==1)))],'k.','LineWidth',2);
ylabel('Raw number of offers accepted')
xlabel('apathy status (no/yes)')
title('Apathetic Patients accept less offers')
 set(gca,'fontSize',14,'fontWeight','bold')
% and also look at cumulative offers accepted

% and plot ~rate of acceptance across experiment for 2 groups
sm=20;%what is the smoothing kernal
figure()
errorBarPlot(smoothn(yesTrial(apVec==0,:),sm),'b','LineWidth',1.5);hold on
errorBarPlot(smoothn(yesTrial(apVec==1,:),sm),'r','LineWidth',1.5);
xlim([0 179])
legend('No Apathy','Apathy')
xlabel('Trial number')
ylabel('Rate of Acceptance')
title('Smoothed (36) response rate for 2 groups across entire experiment')

%and generate MRI regressor
if 0
    mriAccept = grpD.accept(filter)';
    m = mean(mriAccept);
    mriAcc_demean = mriAccept-m;
end
close all
%% *************** RESULTS - Offers accepted - RAW **********************

% ************* Basic results - number of apples gathered, offers accepted
% etc******************************
c = @cmu.colors;
close all
numT=180; % use this to change acceptances to a proportion, if want raw just make it 1.
aaa=accept/180;
color1=[0.5843 0.8157 0.9882];
color2=[0.6350,0.0780,0.1840];
figure();hold on;

b1=bar(1,mean(aaa(apVec==0,1)),'FaceColor',c('dark pastel blue'));hold on;  ...
errorbar(1,mean(aaa(apVec==0,1)),std(aaa(apVec==0,1))./sqrt(length(find(apVec==0))),'Color','k','LineWidth',3)
b2=bar(2,mean(aaa(apVec==1,1)),'FaceColor',c('brick red'));hold on; 
errorbar(2,mean(aaa(apVec==1,1)),std(aaa(apVec==1,1))./sqrt(length(find(apVec==1))),'Color','k','LineWidth',3)

%plot(vecnoA,aaa(apVec==0,:),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5])
%plot(vecA,aaa(apVec==1,:),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5]);
if 0
  
  plot(1*ones(length(accept(apVec==0,1)),1),accept(apVec==0,1),'k.')
  plot(2*ones(length(accept(apVec==1,1)),1),accept(apVec==1,1),'k.')
end
xlim([0.5 2.5])
ylim([0.4 1.5])

ax=gca;
set(ax,'XTick',[1 2],'fontWeight','bold','fontSize',20,'ylim',[.4 1.1],'YTick',[.4:.1:1],'YTickLabel',{'0.4','0.5','0.6','0.7','0.80','0.9','1.0'},'XTicklabel',{'No Apathy','Apathy'});
title('Mean Offers Accepted')

ylabel('Proportion offers accepted (%)')
% ********** Cumulative Acceptance ********************
cumAccept=(zeros(subj,1));
for i = 1:subj
    for t=1:180
        if yesTrial(i,t)==1
            cumAccept(i,t+1) = cumAccept(i,t) + 1;
        else cumAccept(i,t+1) = cumAccept(i,t);
        end
    end
end
% cumAccept_HC=(zeros(subj_HC,1));
% for i = 1:subj_HC
%     for t=1:180
%         if yesTrial_HC(i,t)==1
%             cumAccept_HC(i,t+1) = cumAccept_HC(i,t) + 1;
%         else cumAccept_HC(i,t+1) = cumAccept_HC(i,t);
%         end
%     end
% end

figure();hold on
if 0 % If want to include HC individual cum accept plots - actually this makes things messier so don't use
% for i=1:subj_HC
%     plot(smoothn(cumAccept_HC(i,:)),'m:','LineWidth',3)
% end
end
for i = 1:subj
    if apVec(i)==0
        plot(smoothn(cumAccept(i,:)),'b','LineWidth',3)
    else
        plot(smoothn(cumAccept(i,:)),'r','LineWidth',3)
    end
end
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[0:20:180],'YTick',[0:20:180]);

xlabel('Trial Number');ylabel('Cumulative trials accepted');


% ****************   And plot response rate *****************

sm=36;%what is the smoothing kernal
figure()
%errorBarPlot(smoothn(2,yesTrial_HC,sm),'m','LineWidth',4);
hold on
errorBarPlot(smoothn(yesTrial(apVec==0,:),sm),'b','LineWidth',3);
errorBarPlot(smoothn(yesTrial(apVec==1,:),sm),'r','LineWidth',3);
xlim([18 162]);ylim([0.4 1.2])
legend('No Apathy','Apathy')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'YTick',0.5:0.1:1);
xlabel('Trial number')
ylabel('Acceptance rate (smoothed)')
title('Smoothed (36) response rates')


%% *********** RESULTS - EFFECTS OF REWARD AND EFFORT *************
% Plot raw choice proportions in expanded form
close all
subplot(1,2,1)
for i=1:6
    errorBarPlot(squeeze(choices(i,:,apVec==0))','--','LineWidth',5);hold on
    legend('1','2','3','4','5','6')
    title('non apathetic')
    colormap
end
set(gca,'ColorOrderIndex',1);
subplot(1,2,2)
for i=1:6
    errorBarPlot(squeeze(choices(i,:,apVec==1))',':','LineWidth',5);hold on
end
makeSubplotScalesEqual(1,2);


%%
% generate the RM-ANOVA dataset - use the ARCSINE transformed dataset for
% thesis
temp=[]; % variable to store the choice proportions
tempArc=[];
tempf = [];
tempflog = [];
for i=1:subj % each subject
    temp(i,:)=reshape(choices(:,:,i)',36,1);
    tempArc(i,:)=asin(temp(i,:));
    tempv(i,:)=reshape(vig(:,:,i)',36,1);
    % same for decision time, which I have log transformed. 
    tempt(i,:)=reshape(times(:,:,i)',36,1);

end
    vivec = nanzscore(tempv);
    tempArcz = nanzscore(tempArc);
    temptz   = nanzscore(tempt);
% check if normalised 
close all
figure()
hist(vivec)
figure()
hist(tempArcz)
figure()
hist(tempt)


%% **************** 2D plots ***********************
%using just errorbar function to avoid difficulties with errorBarPlot


depVec=;
for i=1:subj
    if FQs_Ex.BDI(i) > 13 || FQs_Ex.AES_TOTAL(i) > 37 %|| 
      
        depVec(i)=1;
    else
        depVec(i)=0;
    end
end
depVec=depVec';

apVec= apVec29;

figure()
c = @cmu.colors;


%subplot(1,3,2);
dat = squeeze(mean(choices,2))';
H1=shadedErrorBar(1:6,nanmean(dat(apVec==0,:)),   ...
    nanstd(dat(apVec==0,:)./sqrt(length(find(apVec==0)))),'lineprops',...
    {'-.','color', c('air force blue')},...
    'patchSaturation',0.3);
hold on 
H2=shadedErrorBar(1:6,nanmean(dat(apVec==1,:)),   ...
    nanstd(dat(apVec==1,:)./sqrt(length(find(apVec==1)))),'lineprops',...
    {'-.','color', c('brick red')},...
    'patchSaturation',0.3);axis square
ylim([0 1.1]);xlim([0 7])
ax=gca;
set(ax,'fontWeight','bold','fontSize',18,'XTick',[1:1:6], ...
  'XTickLabel',{'10','24','38','52','66','80'})
xlabel('Effort (% MVC)')
ylabel('Prop. Offers Accepted')
ylim([0.2 1])
title('Apathy does not significantly alter effortful behaviour','FontSize',20);
hold off

[lgd, icons, plots, txt] = legend([H1.mainLine H2.mainLine],{'No Apathy','Apathy'});

dat = squeeze(mean(choices,1))';
%subplot(1,3,1)
H1=shadedErrorBar(1:6,nanmean(dat(apVec==0,:)),   ...
    nanstd(dat(apVec==0,:)./sqrt(length(find(apVec==0)))),'lineprops',...
    {'-.','color', c('royal purple')},...
    'patchSaturation',0.3);
hold on 
H2=shadedErrorBar(1:6,nanmean(dat(apVec==1,:)),   ...
    nanstd(dat(apVec==1,:)./sqrt(length(find(apVec==1)))),'lineprops',...
    {'-.','color', c('brick red')},...
    'patchSaturation',0.3);axis square
ylim([0 1.1]);xlim([0 7])
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:6], ...
  'XTickLabel',{'1','3','6','9','12','15'})
xlabel('Reward (Apples)')
ylabel('Prop. Offers Accepted')
ylim([0 1])
title('Apathy significantly reduces reward Incentivisation','FontSize',18);
hold off

[lgd, icons, plots, txt] = legend([H1.mainLine H2.mainLine],{'No Apathy','Apathy'});


%% Apathy - No Apathy plots (raw difference)
%  3D difference plot (2D plot not amazing...
%subplot(1,3,3)
choiceDif=(mean(choices(:,:,apVec==0),3)-mean(choices(:,:,apVec==1),3));
h=surf(choiceDif);shading('interp');hold on;colormap('jet');%colorbar('Ticks',0:.05:.2)
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:6],'YTickLabel',{'1','2','3','4','5','6'},'YTick',[1:1:6],'XTickLabel',{'1','2','3','4','5','6'},'ZTick',[0:0.02:0.16],'ZTickLabel',{'0','0.05','0.1','0.15','0.2'})
%title('3D plot neither vs.either')
%ylabel('Effort (%MVC)')
%xlabel('Reward')
%zlabel('Difference Prop. accepted')
hold on;
base=zeros(6,6);
hh=surf(base);
hh.FaceColor=[0.5 0.5 0.5];hh.FaceAlpha=1;
view(0,30)
xlim([0.5 6.5])
zlim([-0.025 0.225])
if 1 % if want to add on grid lines
    for i=1:5
        plot3(1:6,(i)*ones(6,1),choiceDif(i,1:6),'k:','LineWidth',2)
        plot3((i)*ones(6,1),1:6,choiceDif(1:6,i),'k:','LineWidth',2)
    end
end
colorbar

%ax.ZTickLabel={'0','0.1','0.2','0.3','0.4'}

%% *********************** Computational Models Section  ****************
close all
... here, graph results of model comparison, then present fit data for chosen model (individual and group level)
    ... then split by apathy and show effects...
    
load params_Lin_Quad_Hyp_Exp % contains params for 4 models + AIC and bicf ... order: LIN:Intercept|Interaction|beta only QUAD:Inter|Interaction|beta EXP: Inter|beta HYPER:Inter|beta

figure(); % graph AIC comparison
if 0
    bar(1:3,median(bicf(:,1:3)),'FaceColor',[0.9 0.9 0.9],'EdgeColor',[0.9 0.9 0.9]);hold on
    bar(4:6,median(bicf(:,4:6)),'FaceColor',[0.65 0.65 0.65],'EdgeColor',[0.65 0.65 0.65])
    bar(7:19,median(bicf(:,7:19)),'FaceColor',[0.35 0.35 0.35],'EdgeColor',[0.35 0.35 0.35])
    bar(9:10,median(bicf(:,9:10)),'FaceColor',[0.1 0.1 0.1],'EdgeColor',[0.1 0.1 0.1])
    ax=gca;
    set(ax,'XTick',1:10,'fontWeight','bold','fontSize',20)
    ylabel('bicf');
    xlim([0 11])
else
    bicf(:,5)=[];bicf(:,2)=[];
    bar(1:2,median(bicf(:,1:2)),'FaceColor',[0.9 0.9 0.9],'EdgeColor',[0.9 0.9 0.9]);hold on
    bar(3:4,median(bicf(:,3:4)),'FaceColor',[0.65 0.65 0.65],'EdgeColor',[0.65 0.65 0.65])
    bar(5:6,median(bicf(:,5:6)),'FaceColor',[0.35 0.35 0.35],'EdgeColor',[0.35 0.35 0.35])
    bar(7:19,median(bicf(:,7:19)),'FaceColor',[0.1 0.1 0.1],'EdgeColor',[0.1 0.1 0.1])
    ax=gca;
    set(ax,'XTick',1:19,'fontWeight','bold','fontSize',20)
    ylabel('bicf');
    xlim([0 9])
end

%% CHOSEN MODEL FITS WITH RAW CHOICE DATA
% and now for chosen model plot per subject fits - 2D imagesc probably best
 % rew / eff / beta (where beta is a noise parameter)
clear mod_P model_P mod_V b model_PNL mod_PNL
%;    1=Exp 2=quad 3=quad_beta
close all
model = 1;

if model ==1   % Exponential model
    load b_exp_Cad
    b=b_exp;
    load params_NL_HC % loads b_HC
elseif model ==2   % Quadratic model
    load b_quad_CAD.mat
    b=b_quad;
elseif model ==3  % Quadratic model with beta (noise) parameter instead of intercept
    load params_quad_cad.mat
    b=params_quad_cad;
elseif model==4
    load b_logI_CAD % logistic model with intercept
    b=b_logI;
end


% create est_choice output


for i = 1:19 % each patient
    %b=b(i,:);
    reward=D.stake(i,:)';
    effort=D.effort(i,:)';
    effM = unique(effort);
    rewM = unique(reward);
    %effM = [0.1:0.01:0.80]';
    %rewM = [1:0.2:15]';
    figure(1)
    set(gca,'ColorOrderIndex',1);
    subplot(4,5,i)
    plot(rewM,choices(:,:,i)','o-','LineWidth',3);hold on;
    %v = bsxfun( @plus, b(1)*rew', b(3)*effM));

    if model ==1 % if doing the non-linear model
        if 1
            mod_V(:,:,i) = bsxfun( @times, b(i,2)*rewM', exp(-b(i,3)*effM)) + b(i,1);
            mod_V_HC(:,:,i) = bsxfun( @times, b_HC(i,2)*rewM', exp(-b_HC(i,3)*effM)) + b_HC(i,1);
        else
        for e=1:6
            for r=1:6
  
                mod_V(e,r,i)=b(i,1) + b(i,2)*rewM(r)*exp(-b(i,3)*effM(e));
            end
        end
        end
        for e=1:length(effM)
            for r=1:length(rewM)
                v=mod_V(e,r,i); % value of offer
                v_HC=mod_V_HC(e,r,i);
                mod_PNL(e,r)=exp(v)./(1 + exp(v));
                model_PNL(e,r,i)=exp(v)./(1 + exp(v));
                model_PNL_HC(e,r,i)=exp(v_HC)./(1 + exp(v_HC));
            end
        end
        set(gca,'ColorOrderIndex',1); % keep colours the same for each effort level
        plot(rewM,mod_PNL',':','LineWidth',3);hold off;
        
        
    elseif model ==2
        for e=1:6
            for r=1:6

                mod_V(e,r,i)=b(i,1) + b(i,2)*rewM(r) + b(i,3)*(effM(e)^2);
            end
        end
        
        for e=1:6
            for r=1:6
                v=mod_V(e,r,i); % value of offer
                mod_PQ(e,r)=exp(v)./(1 + exp(v));
                model_PQ(e,r,i)=exp(v)./(1 + exp(v));
            end
        end
        
        set(gca,'ColorOrderIndex',1); % keep colours the same for each effort level
        plot(rewM,mod_PQ,':','LineWidth',3);hold off;
        
    elseif model == 3
        for e=1:6
            for r=1:6

                mod_V(e,r,i)=b(i,1)*rew(r)-b(i,2)*(eff(e)^2);
            end
        end

        % And now transform this into a probability estimate - (done)/(couldhavedone)
        for e=1:6
            for r=1:6
                v=mod_V(e,r,i); % value of offer
                mod_PQb(e,r)=exp(v/b(i,3))./(1 + exp(v/b(i,3)));
                model_PQb(e,r,i)=exp(v/b(i,3))./(1 + exp(v/b(i,3)));
            end
        end
        set(gca,'ColorOrderIndex',1); % keep colours the same for each effort level
        plot(rew,mod_PQb,':','LineWidth',3);hold off;
        
    elseif model ==4
        for e=1:6
            for r=1:6

                mod_V(e,r,i)=b(i,1) + b(i,2)*rew(r) + b(i,3)*(eff(e))+ b(i,4)*rew(r)*eff(e);
            end
        end
        
        for e=1:6
            for r=1:6
                v=mod_V(e,r,i); % value of offer
                mod_PQ(e,r)=exp(v)./(1 + exp(v));
                model_PQ(e,r,i)=exp(v)./(1 + exp(v));
            end
        end
        
        set(gca,'ColorOrderIndex',1); % keep colours the same for each effort level
        plot(rew,mod_PQ',':','LineWidth',3);hold off;
    end
end
    
% And plot 2D heatmaps
figure()
cc=1;
for i=1:19
    subplot(7,6,cc);
    imagesc(flipud(choices(:,:,i)))
    colormap('bone')
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
    subplot(7,6,cc);
    imagesc(flipud(model_PNL(:,:,i)))
    colormap('bone')
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
end

%% or plot heatmaps separating apathy and no apathy subjects
close all
figure()
k=0;
cc=1;

tt=length(find(apVec==k));
temp=find(apVec==k)'; % subjects
for i=1:tt
    subplot(19,2,cc);
    imagesc(flipud(choices(:,:,temp(i))))
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
    axis square
    subplot(19,2,cc);
    imagesc(flipud(model_PNL(:,:,temp(i))))
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
    axis square
end
k=1;


tt=length(find(apVec==k));
temp=find(apVec==k)'; % subjects
for i=1:tt
    subplot(19,2,cc);
    imagesc(flipud(choices(:,:,temp(i))))
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
    axis square
    subplot(19,2,cc);
    imagesc(flipud(model_PNL(:,:,temp(i))))
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
    axis square
end
%%
close all
figure()
k=0;
cc=1;

tt=length(find(apVec==k));
temp=find(apVec==k)'; % subjects
for i=1:tt
%     subplot(19,1,cc);
%     imagesc(flipud(choices(:,:,temp(i))))
%     ax=gca;
%     set(ax,'XTick',[],'YTick',[])
%     cc=cc+1;
    subplot(19,1,cc);
    imagesc(flipud(model_PNL(:,:,temp(i))))
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
end
k=1;


tt=length(find(apVec==k));
temp=find(apVec==k)'; % subjects
for i=1:tt
%     subplot(19,1,cc);
%     imagesc(flipud(choices(:,:,temp(i))))
%     ax=gca;
%     set(ax,'XTick',[],'YTick',[])
%     cc=cc+1;
    subplot(19,1,cc);
    imagesc(flipud(model_PNL(:,:,temp(i))))
    ax=gca;
    set(ax,'XTick',[],'YTick',[])
    cc=cc+1;
end
%% and plot model fits of averaged choice data
close all
% 
figure()
hold on 
%errorBarPlot(squeeze(mean(choices_HC,1))','Color',[0.7 0.7 0.7],'LineWidth',5);hold on
errorBarPlot(squeeze(mean(choices(:,:,apVec==0),1))','b','LineWidth',5);
errorBarPlot(squeeze(mean(choices(:,:,apVec==1),1))','r','LineWidth',5);
xlim([0 7])
ylim([0 1.1])
%errorBarPlot(squeeze(mean(model_PNL_HC,1))','--k','LineWidth',3);hold on
errorBarPlot(squeeze(mean(pp(apVec==0,:,:),1))',':k','LineWidth',3);
errorBarPlot(squeeze(mean(pp(apVec==1,:,:),1))',':k','LineWidth',3);
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:1:6],'XTickLabel',{'1','3','6','9','12','15'})
xlabel('Reward level')
ylabel('Proportion of offers accepted')
axis square
%legend('Controls','SVDc No Apathy','SVDc Apathy')

%% And now plot the parameter estimates - include HCs as well for this
close all
clear b b_HC
... PEs for exponential model are contained within 'b' and b_HC
if 1 % Nonlinear model
load params_NL_HC;% loads beta est for HCs  (intercept|reward|effort)

b_HC(19,:)=[];  % NOTE : HC subject 16 accepted all offers except rejecting one high
% reward one - makes modelling very difficult and model has ~failed
% therefore exclude this subject from analysis of PEs.
load b_exp_Cad; b=b_exp;
if 1
    b(16,:)=[]; % Similarly this subject also accepted all offers therefore exclude at this stage
    ap=apVec;ap(16)=[];
else
    ap=apVec
end
logR = 1; % do you want to take log of reward?
if logR
    b(:,2)=log(b(:,2));
    b_HC(:,2)=log(b_HC(:,2));
end
% else % Quadratic model - THIS is a bit of a mess, so stick with NONLINEAR AS PLANNED
%     load b_quad_CAD.mat
%     b=b_quad;
%     b(:,2)=log(b(:,2));
%     
%     load params_quad_HC % load b_HC which is the betas from 3 PE quad model
%     ap=apVec;
% end

end
% make filter to exclude CAD subject 12 intercept from bar graph (big
% outlier)
filter=ap==0;filter(12)=0;

figure()
bar(1,mean(b_HC(:,1)),'FaceColor',[0.7 0.7 0.7]);hold on;errorbar(1,mean(b_HC(:,1)),std(b_HC(:,1))./sqrt(18),'k','LineWidth',2);
bar(5,mean(b_HC(:,2)),'FaceColor',[0.7 0.7 0.7]);errorbar(5,mean(b_HC(:,2)),std(b_HC(:,2))./sqrt(18),'k','LineWidth',2);
bar(9,mean(b_HC(:,3)),'FaceColor',[0.7 0.7 0.7]);errorbar(9,mean(b_HC(:,3)),std(b_HC(:,3))./sqrt(18),'k','LineWidth',2);
bar(2,mean(b(filter,1)),'b');errorbar(2,mean(b(filter,1)),std(b(filter,1))./sqrt(19),'k','LineWidth',2);
bar(6,mean(b(ap==0,2)),'b');errorbar(6,mean(b(ap==0,2)),std(b(ap==0,2))./sqrt(19),'k','LineWidth',2);
bar(10,mean(b(ap==0,3)),'b');errorbar(10,mean(b(ap==0,3)),std(b(ap==0,3))./sqrt(19),'k','LineWidth',2);
bar(3,mean(b(ap==1,1)),'r');errorbar(3,mean(b(ap==1,1)),std(b(ap==1,1))./sqrt(11),'k','LineWidth',2);
bar(7,mean(b(ap==1,2)),'r');errorbar(7,mean(b(ap==1,2)),std(b(ap==1,2))./sqrt(11),'k','LineWidth',2);
bar(11,mean(b(ap==1,3)),'r');errorbar(11,mean(b(ap==1,3)),std(b(ap==1,3))./sqrt(11),'k','LineWidth',2);
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[2 6 10],'XTickLabel',{'Intercept','Reward','Effort'})
ylabel('Parameter Estimate')
ylim([-6 19])

if 0
figure()
bar(1,mean(b(filter,1)),'b');hold on;errorbar(1,mean(b(filter,1)),std(b(filter,1))./sqrt(19),'k');
bar(4,mean(b(ap==0,2)),'b');errorbar(4,mean(b(ap==0,2)),std(b(ap==0,2))./sqrt(19),'k');
bar(7,mean(b(ap==0,3)),'b');errorbar(7,mean(b(ap==0,3)),std(b(ap==0,3))./sqrt(19),'k');
bar(2,mean(b(ap==1,1)),'r');errorbar(2,mean(b(ap==1,1)),std(b(ap==1,1))./sqrt(11),'k');
bar(5,mean(b(ap==1,2)),'r');errorbar(5,mean(b(ap==1,2)),std(b(ap==1,2))./sqrt(11),'k');
bar(19,mean(b(ap==1,3)),'r');errorbar(19,mean(b(ap==1,3)),std(b(ap==1,3))./sqrt(11),'k');
end








%% ********** extra stats things - ****************
% PLOT RESIDUALS
close all
load cadResid
load cadZResid
figure()
subplot(1,2,1)
hist(reshape(cadResid,684,1));hold on;
title('Residuals from RM-ANOVA - SVDc')
ylabel('count')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20)
subplot(1,2,2)
hist(reshape(cadZResid,684,1));hold on;
title('Z scoredResiduals from RM-ANOVA - SVDc')
ylabel('count')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20)


%% and now work out residuals from exponential model
load params_NL_cad.mat % loads params
modEst=params{1}.prob;
subCh =params{1}.data;
residCad=[];
for k=1:19 % each subject
    tempM = modEst{k}';
    tempC = subCh(k).choices';
    residCad=[residCad;k*ones(length(tempM),1) tempC-tempM];
    clear tempM tempC
    filter=residCad(:,1)==k;
    residM(k)=mean(residCad(filter,2));
end
close all
figure()
hist(residCad(:,2));hold on
title('Residuals from Exp model - SVDc')
ylabel('count')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20)
figure()
hist(residM)








%% **************** Force metrics and Vigor *************************
% Want to plot
... squeeze level at each effort level - mean and all ?
    ... did squeeze vary with apathy (vigor assessment) - could do this as
    ... with PD study, or e.g. vigor(R6)-vigor(R1) for each subject
    
%% force exerted at each level 
% And note, for most part (from C147 handles) conversion factor is
% Right:1kg = 0.03V, Left: 1kg = 0.04V.
%
c = @cmu.colors;

% check magnitude ofpeak squeeze at each force level for the whole group
allforce = squeeze(nanmean(nanmean(grpD.maxforce{1}(:,:,:,:),4),2))';
% check magnitude ofpeak squeeze at each force level for each subgroup
NAp_Sq = squeeze(nanmean(nanmean(grpD.maxforce{1}(:,:,apVec==0,:),4),2))';
Ap_Sq=   squeeze(nanmean(nanmean(grpD.maxforce{1}(:,:,apVec==1,:),4),2))';
%HC_Sq = squeeze(nanmean(nanmean(grpD_HC.maxforce{1},4),2))';
if 1 % limit max force exerted to 1
    NAp_Sq(NAp_Sq>1)=1;
    Ap_Sq(Ap_Sq>1)=1;
    allforce(allforce>1)=1;
 %   HC_Sq(HC_Sq>1)=1;
end

% whole group first 
figure()
%errorbar(nanmean(HC_Sq),nanstd(HC_Sq)./sqrt(19),':','Color',[0.7 0.7 0.7],'LineWidth',5);hold on
H1=shadedErrorBar(1:6,nanmean(allforce),nanstd(allforce)./sqrt(83),'lineprops',...
    {'-.','color', c('dark pastel blue')},...
    'patchSaturation',0.3);

ax=gca;
ylim([0.2 1])
xlim([0 7])
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6],'XTickLabel',{0.1 0.24 0.38 0.52 0.66 0.80})
axis square
[lgd, icons, plots, txt] = legend([H1.mainLine],{'Whole Group'});
xlabel('Required Effort (proportion MVC)');ylabel('Peak squeeze (proportion MVC)');


figure()
%errorbar(nanmean(HC_Sq),nanstd(HC_Sq)./sqrt(19),':','Color',[0.7 0.7 0.7],'LineWidth',5);hold on
H1=shadedErrorBar(1:6,nanmean(NAp_Sq),nanstd(NAp_Sq)./sqrt(length(find(apVec==0))),'lineprops',...
    {'-.','color', c('dark pastel blue')},...
    'patchSaturation',0.3);
hold on
H2=shadedErrorBar(1:6,nanmean(Ap_Sq),nanstd(NAp_Sq)./sqrt(length(find(apVec==1))),'lineprops',...
    {'-.','color', c('brick red')},...
    'patchSaturation',0.3);hold off
ax=gca;
ylim([0.2 1])
xlim([0 7])
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6],'XTickLabel',{0.1 0.24 0.38 0.52 0.66 0.80})
axis square
[lgd, icons, plots, txt] = legend([H1.mainLine H2.mainLine],{'No Apathy','Apathy'});
xlabel('Required Effort (proportion MVC)');ylabel('Peak squeeze (proportion MVC)');

%% Failed squeeze trials
% grpD.failCorr (number of failures divided by number of trials (after
% false squeezes removed)
close all

[aa bb]=ttest2(grpD.failCorr(apVec==0),grpD.failCorr(apVec==1));% ttests show no significant difference between groups
% create random numbers for scatter plot...
for i=1:length(find(apVec==0))
    vecnoA(i) = 1+((rand-0.5)/10);
end
for i=1:length(find(apVec==1))
    vecA(i) = 2+((rand-0.5)/10);
end

for i=1:19
    vecHC(i) = 1+((rand-0.5)/10);
end


% APATHY
figure();
%bar(1,mean(grpD_HC.failCorr),'FaceColor',[0.7 0.7 0.7]);errorbar(1,mean(grpD_HC.failCorr),std(grpD_HC.failCorr)./sqrt(19),'color',[0.5 0.5 0.5],'LineWidth',3);
%plot(vecHC,grpD_HC.failCorr,'k.','MarkerSize',10);
bar(1,mean(grpD.failCorr(apVec==0)),'b'),hold on ;errorbar(1,mean(grpD.failCorr(apVec==0)),std(grpD.failCorr(apVec==0))./sqrt(length(find(apVec==0))),'Color',[0.5 0.5 0.5],'LineWidth',3);
plot(vecnoA,grpD.failCorr(apVec==0),'k.','MarkerSize',10);
bar(2,mean(grpD.failCorr(apVec==1)),'r');errorbar(2,mean(grpD.failCorr(apVec==1)),std(grpD.failCorr(apVec==1))./sqrt(length(find(apVec==1))),'Color',[0.5 0.5 0.5],'LineWidth',3);
plot(vecA,grpD.failCorr(apVec==1),'k.','MarkerSize',10);
ax=gca;
ylabel('Prop. failed trials')
set(gca,'fontWeight','bold','FontSize',20,'XTick',[1 2],'XTickLabel',{})
ylim([0 .16])

% and same plot but just for 2 highest effort levels

figure();hold on
%bar(1,mean(grpD_HC.failHighEff2),'m');errorbar(1,mean(grpD_HC.failHighEff2),std(grpD_HC.failHighEff2)./sqrt(19),'color',[0.5 0.5 0.5],'LineWidth',3);
%plot(vecHC,grpD_HC.failHighEff2,'k.','MarkerSize',10);
bar(1,mean(grpD.failHighEff2(apVec==0)),'b');errorbar(1,mean(grpD.failHighEff2(apVec==0)),std(grpD.failHighEff2(apVec==0))./sqrt(length(find(apVec==0))),'Color',[0.5 0.5 0.5],'LineWidth',3);
plot(vecnoA,grpD.failHighEff2(apVec==0),'k.','MarkerSize',10);
bar(2,mean(grpD.failHighEff2(apVec==1)),'r');errorbar(2,mean(grpD.failHighEff2(apVec==1)),std(grpD.failHighEff2(apVec==1))./sqrt(length(find(apVec==1))),'Color',[0.5 0.5 0.5],'LineWidth',3);
plot(vecA,grpD.failHighEff2(apVec==1),'k.','MarkerSize',10);
ax=gca;
ylabel('Number failed trials (highest 2 effort levels)')
set(gca,'fontWeight','bold','FontSize',20,'XTick',[1 2],'XTickLabel',{'SVD-NoAp','SVD-Apathy'})
%ylim([0 .16])


%% Vigour stuff
% subtract expected...
apVec = apVec34
close all
clear vig
Uneff = [0.1 0.24 0.38 0.52 0.66 0.80];
tempData = nanmean(grpD.maxforce{1},4);
%tempData_HC=nanmean(grpD_HC.maxforce{1},4);
for k=1:subj
    for r=1:6
        for e=1:6
            vig(e,r,k)=tempData(e,r,k)-Uneff(e);
            
            %vig_HC(e,r,k)=tempData_HC(e,r,k)-eff(e);
        end
    end
end

%any difference is excess squeeze between groups? Collapse across R and E
%tempHC=squeeze(nanmean(nanmean(vig_HC)));
tempN=squeeze(nanmean(nanmean(vig(:,:,apVec==0))));
tempA=squeeze(nanmean(nanmean(vig(:,:,apVec==1))));
figure()
bar(1,nanmean(tempN),'b'); hold on; errorbar(1,nanmean(tempN),std(tempN)./sqrt(length(find(apVec==0))),'k','LineWidth',2)
bar(2,nanmean(tempA),'r');errorbar(2,nanmean(tempA),std(tempA)./sqrt(length(find(apVec==1))),'k','LineWidth',2)
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',1:2,'XTickLabel',{'SVD NoA','SVD Ap'})
ylabel('Motor Vigour Ix ')
title('No Difference in Vigour between groups')
% Are there any vigour effects if we compare peak squeeze (at low efforts)
% in the high and low reward categories. ANSWER = NO
%tempHC=squeeze(nanmean(nanmean(vig_HC(1:2,5:6,:))))-squeeze(nanmean(nanmean(vig_HC(1:2,1:2,:))))
tempN=squeeze(nanmean(nanmean(vig(1:2,5:6,apVec==0))))-squeeze(nanmean(nanmean(vig(1:2,1:2,apVec==0))));
tempA=squeeze(nanmean(nanmean(vig(1:2,5:6,apVec==1))))-squeeze(nanmean(nanmean(vig(1:2,1:2,apVec==1))));

figure()
%bar(1,mean(tempHC),'m');hold on;errorbar(1,mean(tempHC),std(tempHC)./sqrt(19),'k','LineWidth',2)
bar(1,nanmean(tempN),'b');hold on;errorbar(1,nanmean(tempN),nanstd(tempN)./sqrt(length(find(apVec==0))),'k','LineWidth',2)
hold on
bar(2,nanmean(tempA),'r');errorbar(2,nanmean(tempA),nanstd(tempA)./sqrt(length(find(apVec==1))),'k','LineWidth',2)
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',1:2,'XTickLabel',{'SVD NoAp','SVD Ap'})
ylabel('Effect of Reward on change in Motor Vigour')

%% Decision Time
% Use decisionTime and decisionTime_HC - have values <0.4 'nan'
% there is an outlier here which I want to remove. 
clear DT ap vecA vecnoA
DT = dt;
DT = nanzscore(log(DT));
ap=apVec34;

close all
for i=1:length(find(ap==0))
    vecnoA(i) = 1+((rand-0.5)/5);
end
for i=1:length(find(ap==1))
    vecA(i) = 2+((rand-0.5)/5);
end
    
figure() % First just plot mean dt for the 3 groups

%bar(1,mean(nanmean(decisionTime_HC,2)),'FaceColor',[0.7 0.7 0.7]);hold on;
bar(1,mean(nanmean(DT(ap==0,:),2)),'FaceColor','b');hold on
bar(2,mean(nanmean(DT(ap==1,:),2)),'FaceColor','r');
if 1
 %   plot(vecHC,nanmean(decisionTime_HC,2),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5]);hold on
    plot(vecnoA,nanmean(DT(ap==0,:),2),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5])
    plot(vecA,nanmean(DT(ap==1,:),2),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5])
end
%errorbar(1,mean(nanmean(decisionTime_HC,2)),std(nanmean(decisionTime_HC,2))./sqrt(19),'k','LineWidth',3);
errorbar(1,mean(nanmean(DT(ap==0,:),2)),std(nanmean(DT(ap==0,:),2))./sqrt(length(find(ap==0))),'k','LineWidth',3);
errorbar(2,mean(nanmean(DT(ap==1,:),2)),std(nanmean(DT(ap==1,:),2))./sqrt(length(find(ap==1))),'k','LineWidth',3)
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',1:2,'XTickLabel',{'SVD-noAp','SVDAp'})
ylabel('Decision time (s)')
xlim([0.5 2.5])
ylim([0 5])
title('Average Decision time is Longer in apathy')


% Now Yes versus No trials  ****************

dtYes=[]; % 1st column = yes, 2nd column = no
%dtYes_HC=[];
filter=yesTrial;filter(isnan(filter))=0;filter=logical(filter);
filter2=yesTrial;filter2(filter2==1)=nan;filter2(~isnan(filter2))=1;filter2(isnan(filter2))=0;filter2=logical(filter2);
for i=1:size(DT,1) % each subject
    dtYes(i,1)=mean(DT(i,filter(i,:)));
    dtYes(i,2)=nanmean(DT(i,filter2(i,:)));
end

figure()
%bar(1,nanmean(dtYes_HC(:,1)),'FaceColor',[0.7 0.7 0.7]);hold on;errorbar(1,nanmean(dtYes_HC(:,1)),nanstd(dtYes_HC(:,1))./sqrt(19),'k','LineWidth',3);
%bar(2,nanmean(dtYes_HC(:,2)),'FaceColor',[0.3 0.3 0.3]);hold on;errorbar(2,nanmean(dtYes_HC(:,2)),nanstd(dtYes_HC(:,2))./sqrt(19),'k','LineWidth',3)
h1 = bar(1,nanmean(dtYes(ap==0,1)),'FaceColor',[0.7 0.7 0.7]);hold on;errorbar(1,nanmean(dtYes(ap==0,1)),nanstd(dtYes(ap==0,1))./sqrt(length(find(ap==0))),'k','LineWidth',3); 
h2 = bar(2,nanmean(dtYes(ap==0,2)),'FaceColor',[0.3 0.3 0.3]);hold on;errorbar(2,nanmean(dtYes(ap==0,2)),nanstd(dtYes(ap==0,2))./sqrt(length(find(ap==1))),'k','LineWidth',3); 
h3 = bar(4,nanmean(dtYes(ap==1,1)),'FaceColor',[0.7 0.7 0.7]);hold on;errorbar(4,nanmean(dtYes(ap==1,1)),nanstd(dtYes(ap==1,1))./sqrt(length(find(ap==0))),'k','LineWidth',3);
h4 = bar(5,nanmean(dtYes(ap==1,2)),'FaceColor',[0.3 0.3 0.3]);hold on;errorbar(5,nanmean(dtYes(ap==1,2)),nanstd(dtYes(ap==1,2))./sqrt(length(find(ap==1))),'k','LineWidth',3);
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1.5 4.5],'XTickLabel',{})
ylabel('Decision time (s)');
xlim([0 6]);
%ylim([0 3]);
xticklabels({'NoAP','Ap'});
legend([h1 h2],{'Yes','No'});
title('Decision time vs Trial Type')


%plot this as a normal plot instead of a bar plot
close all
figure()
%bar(1,nanmean(dtYes_HC(:,1)),'FaceColor',[0.7 0.7 0.7]);hold on;errorbar(1,nanmean(dtYes_HC(:,1)),nanstd(dtYes_HC(:,1))./sqrt(19),'k','LineWidth',3);
%bar(2,nanmean(dtYes_HC(:,2)),'FaceColor',[0.3 0.3 0.3]);hold on;errorbar(2,nanmean(dtYes_HC(:,2)),nanstd(dtYes_HC(:,2))./sqrt(19),'k','LineWidth',3)
errorBarPlot(dtYes(ap==0,:),'Color',color1,'LineWidth',3);
hold on 
errorBarPlot(dtYes(ap==1,:),'Color',color2,'LineWidth',3);
legend('SVD No Apathy','SVD Apathy')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1 2],'XTickLabel',{'Yes','No'})
ylabel('(log)Decision time (s)');
%ylim([-1 1])
xlim([0.5 2.5])
title('(log) Decision time vs Choice(Y/N)')

% and for transformed data
close all
figure()
%bar(1,nanmean(dtYes_HC(:,1)),'FaceColor',[0.7 0.7 0.7]);hold on;errorbar(1,nanmean(dtYes_HC(:,1)),nanstd(dtYes_HC(:,1))./sqrt(19),'k','LineWidth',3);
%bar(2,nanmean(dtYes_HC(:,2)),'FaceColor',[0.3 0.3 0.3]);hold on;errorbar(2,nanmean(dtYes_HC(:,2)),nanstd(dtYes_HC(:,2))./sqrt(19),'k','LineWidth',3)
errorBarPlot(dtYes(ap==0,:),'Color',color1,'LineWidth',3);
hold on 
errorBarPlot(dtYes(ap==1,:),'Color',color2,'LineWidth',3);
legend('SVD No Apathy','SVD Apathy')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1 2],'XTickLabel',{'Yes','No'})
ylabel('(log)Decision time (s)');
ylim([1 3])
xlim([0.5 2.5])
title('(log) Decision time vs Choice(Y/N)')



% And finally just plot the difference - YES - NO ie a within subject
% comparison rather than average time plots
figure()
temp=dtYes(:,2)-dtYes(:,1);%tempHC=dtYes_HC(:,2)-dtYes_HC(:,1);
%bar(1,nanmean(tempHC),'m');hold on;errorbar(1,nanmean(tempHC),nanstd(tempHC)./sqrt(19),'k','LineWidth',3);
bar(1,nanmean(temp(ap==0)),'Facecolor','b');hold on;errorbar(1,nanmean(temp(ap==0)),nanstd(temp(ap==0))./sqrt(length(find(ap==0))),'k','LineWidth',3);
bar(2,nanmean(temp(ap==1)),'Facecolor','r');hold on;errorbar(2,nanmean(temp(ap==1)),nanstd(temp(ap==1))./sqrt(length(find(ap==1))),'k','LineWidth',3);
if 1
    %plot(vecHC,nanmean(tempHC,2),'.','MarkerSize',12,'MarkerEdgeColor',[0.5 0.5 0.5])
    plot(vecnoA,nanmean(temp(ap==0,:),2),'.','MarkerSize',12,'MarkerEdgeColor',[0.5 0.5 0.5])
    plot(vecA,nanmean(temp(ap==1,:),2),'.','MarkerSize',12,'MarkerEdgeColor',[0.5 0.5 0.5])
    if 1
        ylim([-1 1.5]);plot(3,-.93,'.','MarkerSize',12,'MarkerEdgeColor',[0.5 0.5 0.5]);
    end
end
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1 2],'XTickLabel',{'noAp','Ap'})
ylabel('Difference in decision time NO-YES (s)');



%% NOW look at how DT relates to value of offer - to try and show all groups sensitive to this.
close all
tempDec=nanmean(log(grpD.decisiont{1}),4);
easyDec=[];
hardDec=[];
for i=1:subj % each subject
    temp=[];
    temp2=[];
    c=1; %counter
    cc=1;
    for e=1:6
        for r=1:6
            if choices(e,r,i) <=0.25 || choices(e,r,i) >=0.75
                temp(c,1)=tempDec(e,r,i);
                c=c+1;
            else
                temp2(cc,1)=tempDec(e,r,i);
                cc=cc+1;
            end
        end
    end
    easyDec(i,1)=nanmean(temp);
    easyDec(i,2)=nanmean(temp2);
end

%tempDec=nanmean(grpD_HC.decisiont{1},4);
%easyDecHC=[];
%hardDecHC=[];
for i=1:subj % each subject
    temp=[];
    temp2=[];
    c=1; %counter
    cc=1;
    for e=1:6
        for r=1:6
            if choices(e,r,i) <=0.25 || choices(e,r,i) >=0.75
                temp(c,1)=tempDec(e,r,i);
                c=c+1;
            else
                temp2(cc,1)=tempDec(e,r,i);
                cc=cc+1;
            end
        end
    end
   % easyDecHC(i,1)=nanmean(temp);
    %easyDecHC(i,2)=nanmean(temp2);
end



figure()
%errorBarPlot(easyDecHC,'Color',[0.7 0.7 0.7],'LineWidth',3);hold on
errorBarPlot(easyDec(ap==0,:),'Color','b','LineWidth',3);
hold on 
errorBarPlot(easyDec(ap==1,:),'Color','r','LineWidth',3);
legend('SVD No Apathy','SVD Apathy')
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1 2],'XTickLabel',{'Easy','Hard'})
ylabel('Decision time (s)');
%ylim([1 3])
xlim([0.5 2.5])


%% ******************** Block Effects   *************************
% grpD.choiceMap has already had accidental choices removed (nan)
close all
figure();hold on
for i=1:5 % for each block
    errorBarPlot(squeeze(nanmean(grpD.choicemap{1}(:,:,apVec==0,i),1))','LineWidth',3);
end
legend('Block 1','Block 2','Block 3','Block 4','Block 5');
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6])
ylim([0.2 1.1]);xlim([0 7]);
title('Non-Ap vs Block')

figure();hold on
for i=1:5 % for each block
    errorBarPlot(squeeze(nanmean(grpD.choicemap{1}(:,:,apVec==1,i),1))','LineWidth',3);
end
legend('Block 1','Block 2','Block 3','Block 4','Block 5');
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6])
ylim([0.1 1.1]);xlim([0 7]);
%figure();hold on
%for i=1:5 % for each block
 %   errorBarPlot(squeeze(nanmean(grpD_HC.choicemap{1}(:,:,:,i),1))','LineWidth',3);
%end
legend('Block 1','Block 2','Block 3','Block 4','Block 5');
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6])
ylim([0.1 1.1]);xlim([0 7]);
title('Ap vs blocks')

%% ******************** Depression Effects   *************************
% create this vector to look at the effect of removing the Depression
% outlier you can place this into the dat equation instead of choices. 
choices_outlier = choices(:,:,[1:46 48:83]);
BDI_outlier = BDI([1:46 48:83]);
AES_outlier = AES_total([1:46 48:83]);
close all

depVec=[];
for i=1:subj
    if FQs_Ex.AES_TOTAL(i) < 37 & FQs_Ex.BDI(i) > 13
      
        depVec(i)=1;
    else
        depVec(i)=0;
    end
end
depVec=depVec';



for i=1:2
    figure();
    dat = squeeze(mean(choices,i))';
   % dat_hc=squeeze(mean(choices_HC,i))';
    %errorbar(1:6,mean(dat_hc),std(dat_hc)./sqrt(19),'m--','LineWidth',3); hold on;
    errorbar(1:6,mean(dat(depVec==0,:)),std(dat(depVec==0,:))./sqrt(length(find(depVec==0))),'b','LineWidth',3); hold on;
    errorbar(1:6,mean(dat(depVec==1,:)),std(dat(depVec==1,:))./sqrt(length(find(depVec==1))),'r','LineWidth',3)
    %title('proportion of offers accepted as reward level increases')
    legend('SVD without Ap+DEp','SVD with AP+Dep ');
    axis square
    ylim([0 1.1]);xlim([0 7])
    ax=gca;
    if i==2
        set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:1:6],'XTickLabel',{'10','24','38','52','66','80'})
        xlabel('Effort level (% MVC)')
        ylabel('Proportion of offers accepted')
    else
        set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:1:6],'XTickLabel',{'1','3','6','9','12','15'})
        xlabel('Reward level')
        ylabel('Proportion of offers accepted')
    end
    title('Apathetic+Depressed')
end
%% Dysphoria subscale
%load bdi_full_cad
bdi_full=xxx;
qq=[1 2 3 5 6 7 8 19 9 10 11 14]; % dysphoria subscale

%BDI_dys_hc=[];
for i=1:size(bdi_full,1)
    BDI_dys(i)=sum(bdi_full(i,qq));
end

%% *************** Generating MR regressors... use this as have changed order of subjects for imaging analysis!!! *****************

% mrFilter contains final 16 SVD subjects in order 
load mrFilter; % this is the order of subjects we want to end up with
mrSub = [1 4:18]'; % this is the subjects included
load Mr_data_Transform_Vector; % this is the vector which transforms subject order into correct one for imaging 
... analyses when sorted (called "yy")
%get age
load demo_Cad19
ages=t.age(1:19); ages=ages(mrSub); 
temp=[yy ages];
temp2 = sortrows(temp,1);
agesMR = temp2(:,2)
clear temp temp2
% plus need to add in the HCs - note excluded subjects as 1, 7 and 18.
conMR=t.age(20:end);filter=[1;7;18];%subjects to exclude
conMR(filter)=[];
mrAges=[conMR;agesMR];mrAges=zscore(mrAges)
%%
conMRgds=t.gdsD(20:end);filter=[1;7;18];%subjects to exclude
conMRgds(filter)=[];



%% And note now need to perfectly match controls to subjects:

%% Playing with the model parameters to get good sense of their effects
% use population median as baseline and then just change one at a time

b = [-19   3   3]; %Non linear


eff = unique(effort);
effM=[0.1:0.01:0.80]';
rew = unique(reward);
rewM= [1:0.1:15]';
i=2;
if 1     
v = b(1) + bsxfun( @times, b(2)*rewM', exp(-b(3)*eff));
            pp=exp(v)./(1 + exp(v));
else
v = b(1) + bsxfun( @plus, b(2)*rew', b(3)*(effM.^2));
            pp=exp(v)./(1 + exp(v));
end
subplot(1,2,1)
imagesc(pp)
colorbar
subplot(1,2,2)
%errorBarPlot(pp)
plot(pp')


%%
%% Set up for statistics - GLME models
%% Hierachical linear mixed effects model - use fitglme -
% move data into subData - [subject Choice Force RT rew eff ap(binary) ...
% AES LARS PCA1(AES-BDI) PCA2(LARS-BDI) PCA3(AES-LARS) etc. depending on
% results. 
% stake, effort and Yestrial within the D array should already have had the practise block removed (as long as above parts run)
subData=[];
linear=0;
% I also want to create a categorical variable for post hoc analysis
% looking specifically at reward level by category. This would possibly


%zscores for qs.
% AES and LARS z scored
AES_total = FQs_Ex.AES_TOTAL;
LARS_Total = FQs_Ex.LARS_TOTAL;
BDI = FQs_Ex.BDI;
D.Index=repmat(1:180,83,1);
% R_F1 = EFA_R_F1;
%%%%%%%


for i=1:subj % each subject
    choicesVec = D.Yestrial(i,:)';
    choicesVec(isnan(choicesVec)) = 0; %change nans to 0
    reward = D.stake(i,:)';
    effort = D.effort(i,:)';
    decVec = dt(i,:)'; %pick individual subjects' dec times
    Z_decVec = nanzscore(log(dt(i,:)'));
    %decVec(isnan(decVec))=0;
    forceVec = D.maximumForce(i,:)';
    forceVec(forceVec<0) = nan;
    vigVec = D.vigour(i,:)';
    block = D.block(i,:)';
    Index = D.Index(i,:)';
    % create a matrix of decision times that will be excluded from
    % the final decision matrix that corresponds to those DT which
    % are more than 3 STD from the mean for that particular patient
    dtind = dt(i,:)';
    RT_Slow = abs(dtind - nanmean(dtind)) >= 3.*nanstd(dtind);
    a=1;
    for j = 1:length(decVec) %for each of the 180 trials
        if decVec(j) < 0.4 || decVec(j) >= mean(decVec)+ ...
                3*nanstd(decVec);
            removal(a) = j; %create vector of trials to remove
            a=a+1;
        end
    end
    if exist('removal')
        choicesVec(removal)= [];
        reward(removal)    = [];
        effort(removal)    = [];
        decVec(removal)    = [];
        forceVec(removal)  = [];
        vigVec(removal)    = [];
        Z_decVec(removal)  = [];
        block(removal)     = [];
        Index(removal)  = [];
    end
    clear removal
    if ~linear
        subData = [subData;i*ones(length(choicesVec),1) choicesVec, ...
            vigVec decVec Z_decVec reward effort                    ...
            apVec29(i)*ones(length(choicesVec),1),                  ...
            AES_total(i)*ones(length(choicesVec),1),                ...
            LARS_Total(i)*ones(length(choicesVec),1),               ...
            Composite(i)*ones(length(choicesVec),1),                ...
            BDI(i)*ones(length(choicesVec),1),                      ...
            RT_Slow(i)*ones(length(choicesVec),1),                  ...                                                     
            block Index apVec34(i)*ones(length(choicesVec),1),      ...
            apVec37(i)*ones(length(choicesVec),1),                  ...
            rank.pAESInv(i)*ones(length(choicesVec),1),             ...
            rank.pBDInv(i)*ones(length(choicesVec),1),...
            apstanres(i)*ones(length(choicesVec),1),...
            F1score(i)*ones(length(choicesVec),1),...
            F2score(i)*ones(length(choicesVec),1),...
            Resapn(i)*ones(length(choicesVec),1),...
            Resdepn(i)*ones(length(choicesVec),1),...
            apMedian(i)*ones(length(choicesVec),1),...
            depMedian(i)*ones(length(choicesVec),1),...
            dep14(i)*ones(length(choicesVec),1),...
            dep19(i)*ones(length(choicesVec),1),...
            apVec365(i)*ones(length(choicesVec),1),...
            apVec40(i)*ones(length(choicesVec),1),...

            ];
        
    end
end

subject   = categorical(subData(:,1));
choice    = subData(:,2);
Vigour    = subData(:,3);
DT        = subData(:,4);
Z_DT      = subData(:,5);
rew       = subData(:,6);
eff       = subData(:,7);
ap29     = subData(:,8);
AES_T     = subData(:,9);
LARS_T    = subData(:,10);
CompAp    = subData(:,11);
Depression= subData(:,12);
RT_slow   = subData(:,13);
Block     = (subData(:,14));
Index     = (subData(:,15));
ap34      = (subData(:,16));
ap37      = (subData(:,17));
apinv     = (subData(:,18));
depinv    = (subData(:,19));
residualap = (subData(:,20));
eff2 = eff.^2;
eff3 = eff.^3;
EFA_Ap = (subData(:,21));
EFA_Dep = (subData(:,22));
Apathy_Residuals = (subData(:,23));
Depression_Residuals = (subData(:,24));
apMedian = (subData(:,25));
depMedian = (subData(:,26));
dep14 = (subData(:,27));
dep19= (subData(:,28));
apathy365 = (subData(:,29));
ap40 = (subData(:,30));
if 1
    rew=nanzscore(rew);
    eff=nanzscore(eff);
    ap34 = nanzscore(ap34);
    ap29 = nanzscore(ap29);
    ap37 = nanzscore(ap37);
    Block = nanzscore(Block);
    AES_T = nanzscore(AES_T);
    LARS_T = nanzscore(LARS_T);
    Depression = nanzscore(Depression);
    Vigour = nanzscore(Vigour);
    DT = nanzscore(DT);
    Index = nanzscore(Index);
    eff2 = nanzscore(eff2);
    eff3 = nanzscore(eff3);
    residualap = nanzscore(residualap);
    EFA_Ap = nanzscore(EFA_Ap);
    EFA_Dep = nanzscore(EFA_Dep);
    Apathy_Residuals = nanzscore(Apathy_Residuals);
    Depression_Residuals = nanzscore(Depression_Residuals);
    MedianAp = nanzscore(apMedian);
    MedianDep = nanzscore(depMedian);
    Dep14 = nanzscore(dep14);
    Dep19 = nanzscore(dep19);
    ap40 = nanzscore(ap40);
%     Factor1 = nanzscore(Factor1);
end
%force=nanzscore(force);
%lars = zscore(lars);

Design = table(choice,Vigour,Z_DT,DT,rew,eff,ap34,ap29,ap37,AES_T,CompAp,Depression, ...
    subject,Block,Index,eff2,eff3,apinv,depinv,residualap,EFA_Ap,EFA_Dep,...
    Apathy_Residuals, Depression_Residuals, MedianAp,MedianDep,Dep14,...
    Dep19,apathy365,ap40);


%% all different model possibilities ( more than 30). 
    
    
% % Full factorial models first. These include interaction terms which I
% % think will have to come out inevitably. 
% 
% %First Total AES as a continuous variable. 
% 'choice ~ rew*eff*AES_T    + rew*eff*Depression + rew*eff*Block   +  (1|subject)'
% % then the composite apathy variable that I created from AES and LARS total
% % scores. This is called CompAp
% 'choice ~ rew*eff*CompAp   + rew*eff*Depression + rew*eff*Block   +  (1|subject)'
% % Then the median split cut off of my composite apathy score. This is
% % called ap. 
% 'choice ~ rew*eff*ap       + rew*eff*Depression + rew*eff*Block   +  (1|subject)'
% % finally the LARS total, which was very noisy. 
% 'choice ~ rew*eff*LARS_T   + rew*eff*Depression + rew*eff*Block   +  (1|subject)'
% 
% 
% % The next three models does not account for block effects. It uses the
% % same three apathy scores used above. AES first. 
% 'choice ~ rew*eff*AES_T    + rew*eff*Depression +  (1|subject)'
% % Composite apathy score next. 
% 'choice ~ rew*eff*CompAp   + rew*eff*Depression +  (1|subject)'
% % Apathy cut off (median split)
% 'choice ~ rew*eff*ap       + rew*eff*Depression +  (1|subject)'
% % Then the lars Total
% 'choice ~ rew*eff*LARS_T   + rew*eff*Depression +  (1|subject)'
% 
% 
% % Now look at models that do not include depression to compare model fits. 
% 'choice ~ rew*eff*AES_T  + rew*eff*Block  +  (1|subject)'
% 'choice ~ rew*eff*CompAp + rew*eff*Block +  (1|subject)'
% 'choice ~ rew*eff*ap     + rew*eff*Block     +  (1|subject)'
% 'choice ~ rew*eff*LARS_T + rew*eff*Block +  (1|subject)'
% % now just look at apathy
% 'choice ~ rew*eff*AES_T  +  (1|subject)'
% 'choice ~ rew*eff*CompAp +  (1|subject)'
% 'choice ~ rew*eff*ap     + (1|subject)'
% 'choice ~ rew*eff*LARS_T +  (1|subject)'
% 'choice ~ rew*eff*Block     + (1|subject)'
% % What about one that includes depression only. This to see how well
% % depression independently explains the data. 
% 'choice ~ rew*eff*Depression + (1|subject)'
% 
% % how about simpler one way effects 
% 'choice ~ rew*eff + (1|subject)'
% 
% 'choice ~ AES_T + (1|subject)'
% 'choice ~ Depression + (1|subject)'
% 
% 
% %Models without interactions
% 'choice ~ AES_T*rew  + AES_T*eff +  (1|subject)'
% 'choice ~ AES_T*rew  + AES_T*eff + Depression*rew + Depression*eff+(1|subject)'
% 'choice ~ Depression*rew + Depression*eff + (1|subject)'
% 
% % new types of models which include a random effect based on trial. 
% 'choice ~ AES_T*rew*eff + (1 + rew:Index + eff:Index + Index + rew:eff|subject)'
% 'choice ~ AES_T*rew*eff + (1 + rew:Index + eff:Index + Index |subject)'
% 'choice ~ AES_T*rew*eff + Depression + (1 + rew:Index + eff:Index + Index + rew:eff|subject)'
% 'choice ~ AES_T*rew*eff + Depression + (1 + rew:Index + eff:Index + Index|subject)'
% 'choice ~ AES_T*rew  + AES_T*eff + Depression + (1+Index:rew+Index:eff+Index|subject)'
% 
% 
% % a simplified version of the above 
% 'choice ~ AES_T*rew*eff + (1 + Index |subject)'
% 'choice ~ AES_T*rew+ AES_T*eff + (1 + Index |subject)'
% 'choice ~ rew+eff + (rew+eff|subject)'
% 'choice ~ rew*eff + (rew*eff|subject)'
clear aic glme_fit bic
linear=0; % which model type to run
models = {
% 'choice ~ rew*eff2*MedianAp +(1|subject)'
% 'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep + (1|subject)'
% 'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep + (1+rew+eff2|subject)'
% 'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep - rew:eff2:MedianAp - rew:eff2:MedianDep + (1|subject)'
% 'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep - rew:eff2:MedianAp - rew:eff2:MedianDep + (1+rew+eff2|subject)'
% 'choice ~ rew*eff2*MedianDep + (1|subject)'   
'choice ~ rew*eff2*Depression_Residuals +(1|subject)'
'choice ~ rew*eff2*Depression_Residuals + rew*eff2*Depression + (1|subject)'
'choice ~ rew*eff2*Depression_Residuals + rew*eff2*AES_T + (1|subject)'
'choice ~ rew*eff2*Depression_Residuals - rew:eff2:Depression_Residuals + rew*eff2*AES_T - rew:eff2:AES_T + (1|subject)'
'choice ~ rew*eff2*Apathy_Residuals +(1|subject)'
'choice ~ rew*eff2*Apathy_Residuals + rew*eff2*Depression + (1|subject)'
'choice ~ rew*eff2*Apathy_Residuals + rew*eff2*AES_T + (1|subject)'
'choice ~ rew*eff2*residualap +(1|subject)'
'choice ~ rew*eff2*residualap + rew*eff2*depinv + (1|subject)'
'choice ~ rew*eff2*residualap + rew*eff2*apinv + (1|subject)'
'choice ~ rew*eff2*apinv  +   (1|subject)'   
'choice ~ rew*eff2*depinv  +  (1|subject)'
'choice ~ rew*eff2*apinv   + rew*eff2*depinv +  (1|subject)'
'choice ~ rew*eff2*EFA_Ap  +   (1|subject)'   
'choice ~ rew*eff2*EFA_Dep  +  (1|subject)'
'choice ~ rew*eff2*EFA_Ap   + rew*eff2*EFA_Dep +  (1|subject)'
'choice ~ rew*eff2*AES_T  +   (1|subject)'   
'choice ~ rew*eff2*Depression  +  (1|subject)'
'choice ~ rew*eff2*AES_T   + rew*eff2*Depression +  (1|subject)'
'choice ~ rew*eff2*AES_T   +  (1|subject)'
'choice ~ rew*eff*AES_T    +  (1|subject)'

};

if linear
    for i=1:length(models)
        sprintf('starting model %g',i)
        glme_fit{i}=fitglme(Design,models{i},'Distribution','normal');
        aic(i)=glme_fit{i}.ModelCriterion.AIC;
        bic(i)=glme_fit{i}.ModelCriterion.BIC;
    end
else
    for i=1:length(models)
        sprintf('starting model %g',i)
        glme_fit{i}=fitglme(Design,models{i},'Distribution','binomial','fitmethod','Laplace');
        aic(i)=glme_fit{i}.ModelCriterion.AIC;
        bic(i)=glme_fit{i}.ModelCriterion.BIC;
                

    end
end
aic=aic-min(aic);
bic=bic-min(bic);

if 1
    save('glme_fit_lin_cat','models','glme_fit')
end
if 0 % save outputs
save('glme_fitPD_Z','models','glme_fit')
end

%% I now want to run this model for the experimental candidates. 
clear aic_cuts glme_fit_cuts bic_cuts
linear = 0; % which model type to run
models_cuts = {
    
'choice ~ rew*eff2*MedianAp +(1|subject)'
'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep + (1|subject)'
'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep + (1+rew+eff2|subject)'
'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep - rew:eff2:MedianAp - rew:eff2:MedianDep + (1|subject)'
'choice ~ rew*eff2*MedianAp + rew*eff2*MedianDep - rew:eff2:MedianAp - rew:eff2:MedianDep + (1+rew+eff2|subject)'
'choice ~ rew*eff2*MedianDep + (1|subject)'   

'choice ~ rew*eff2*ap29 +(1|subject)'
'choice ~ rew*eff2*ap29 + rew*eff2*MedianDep + (1|subject)'
'choice ~ rew*eff2*ap29 + rew*eff2*MedianDep + (1+rew+eff2|subject)'
'choice ~ rew*eff2*ap29 + rew*eff2*MedianDep - rew:eff2:ap29 - rew:eff2:MedianDep + (1|subject)'
'choice ~ rew*eff2*ap29 + rew*eff2*MedianDep - rew:eff2:ap29 - rew:eff2:MedianDep + (1+rew+eff2|subject)'

'choice ~ rew*eff2*ap37 +(1|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep14 + (1|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep14 + (rew*eff2|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep14 - rew:eff2:ap37 - rew:eff2:Dep14 + (1|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep14 - rew:eff2:ap37 - rew:eff2:Dep14 - rew:eff2 + (1+rew+eff2|subject)'
'choice ~ rew*eff2*Dep14 + (1|subject)'   

'choice ~ rew*eff2*ap37 + rew*eff2*Dep19 + (1|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep19 + (rew*eff2|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep19 - rew:eff2:ap37 - rew:eff2:Dep19 + (1|subject)'
'choice ~ rew*eff2*ap37 + rew*eff2*Dep19 - rew:eff2:ap37 - rew:eff2:Dep19 + (1+rew+eff2|subject)'
'choice ~ rew*eff2*Dep19 + (1|subject)'

% which apathy cut off fits best 
'choice ~ rew*eff2*ap37 +(1|subject)'
'choice ~ rew*eff2*ap34 +(1|subject)'
'choice ~ rew*eff2*ap29 +(1|subject)'
'choice ~ rew*eff2*MedianAp +(1|subject)'


%which depression cut off is best. 
'choice ~ rew*eff2*Dep14 +(1|subject)'
'choice ~ rew*eff2*Dep19 +(1|subject)'
'choice ~ rew*eff2*MedianDep +(1|subject)'

%which combination is best
'choice ~ rew*eff2*Dep14 +rew*eff2*ap34 +(1|subject)';
'choice ~ rew*eff2*Dep19 +rew*eff2*ap34 +(1|subject)';
'choice ~ rew*eff2*MedianAp +rew*eff2*MedianDep +(1|subject)';
'choice ~ rew*eff*MedianAp + rew*eff2*Dep19 + (1|subject)';
'choice ~ rew*eff*MedianAp + rew*eff2*Dep14 + (1|subject)';
'choice ~ rew*eff2*MedianAp +rew*eff2*MedianDep +(rew*eff|subject)';
'choice ~ rew*eff2*apathy365 +(1|subject)';
'choice ~ rew*eff2*apathy365 + rew*eff2*Dep14+(1|subject)';
'choice ~ rew*eff2*ap40 +(1|subject)';
'choice ~ rew*eff2*ap40 + rew*eff2*Dep14+(1|subject)';

}





if linear
    for i=1:length(models_cuts)
        sprintf('starting model %g',i)
        glme_fit_cuts{i}=fitglme(Design,models_cuts{i},'Distribution','normal');
        aic_cuts(i)=glme_fit_cuts{i}.ModelCriterion.AIC;
        bic_cuts(i)=glme_fit_cuts{i}.ModelCriterion.BIC;
    end
else
    for i=1:length(models_cuts)
        sprintf('starting model %g',i)
        glme_fit_cuts{i}=fitglme(Design,models_cuts{i},'Distribution','binomial','fitmethod','Laplace');
        aic_cuts(i)=glme_fit_cuts{i}.ModelCriterion.AIC;
        bic_cuts(i)=glme_fit_cuts{i}.ModelCriterion.BIC;
                

    end
end
aicm_cuts=aic_cuts-min(aic_cuts);
bicm_cuts=bic_cuts-min(bic_cuts);


% 'choice ~ rew+eff2 + (rew+eff2|subject)'
% 'choice ~ rew*eff2 + (rew*eff2|subject)'
  
%plot aic of all cut offs 
close;bar(aic_cuts([23 24 26]));ylim([7430 7450]);
xticklabels({'AES37','AES34','AES32-Median'});
ylabel('AIC values for different Cut-Offs');
title('Best Model fit belongs to AES-32')
set(findall(gcf,'type','text'),'FontSize',22,'fontWeight','bold')

if 1
    save('glme_cuts','models_cuts','glme_fit_cuts')
end

%% view p values from all models, for a given effect
effect = '^rew:(AES_T|CompAp|LARS_T|ap)$';
effect = '^rew:eff$';
cellfun( @(x) x.Coefficients.pValue( find(cellfun(@any,regexp(x.CoefficientNames,effect))) ) , glme_fit_cuts , 'uni',0)



%% Analysis for Force. 
clear  glme_fit_force bicf aicf

%assign linearity
linear = 1;

% create models
models_force = {
 
'Vigour ~ rew*eff*AES_T + (1|subject)'
'Vigour ~ rew*eff*ap34 + (1|subject)'


  
  };

  
if linear
    for i=1:length(models_force)
        sprintf('starting model %g',i)
        glme_fit_force{i}=fitglme(Design,models_force{i},'Distribution','normal');
        aicf(i)=glme_fit_force{i}.ModelCriterion.AIC;
        bicf(i)=glme_fit_force{i}.ModelCriterion.BIC;
    end
else
    for i=1:length(models_force)
        sprintf('starting model %g',i)
        glme_fit_force{i}=fitglme(Design,models_force{i},'Distribution','binomial','fitmethod','Laplace');
        aicf(i)=glme_fit_force{i}.ModelCriterion.AIC;
        bicf(i)=glme_fit_force{i}.ModelCriterion.BIC;
    end
end
aicf=aicf-min(aicf);
bicf=bicf-min(bicf);
if 0
    save('glme_fit_lin_cat','models_force','glme_fit_force')
end
if 1 % save outputs
save('glme_fitPD_Z','models_force','glme_fit_force')
end

%% Analysis for decision time. 
clear aic glme_fit_time bict aict

%assign linearity
linear = 1;

% create models
models_time = {
 
'Z_DT ~ rew*eff*AES_T + (1|subject)'
'Z_DT ~ rew*eff2*AES_T + (1|subject)'
'Z_DT ~ rew*eff2*ap34 + (1|subject)'


  };

  
if linear
    for i=1:length(models_time)
        sprintf('starting model %g',i)
        glme_fit_time{i}=fitglme(Design,models_time{i},'Distribution','normal');
        aict(i)=glme_fit_time{i}.ModelCriterion.AIC;
        bict(i)=glme_fit_time{i}.ModelCriterion.BIC;
    end
else
    for i=1:length(models_time)
        sprintf('starting model %g',i)
        glme_fit_time{i}=fitglme(Design,models_time{i},'Distribution','binomial','fitmethod','Laplace');
        aict(i)=glme_fit_time{i}.ModelCriterion.AIC;
        bict(i)=glme_fit_time{i}.ModelCriterion.BIC;
    end
end
aict=aict-min(aict);
bict=bict-min(bict);
if 0
    save('glme_fit_lin_cat','models_time','glme_fit_time')
end
if 1 % save outputs
save('glme_fitPD_Z','models_time','glme_fit_time')
end


%% Abbreviated version of Computational modelling.
% First we To calculate the values for Intrinsic motivation, reward ...
%sensitivity and effort sensitivity fir each patient. 

% First use the appropriate glmemodel
model = { ...
    'choice ~ rew*eff2 + (rew*eff2|subject)'
    };

% Then run the GLME with the two main outputs being B and BNames which
% correspond to the random effects parameters and their corresponding names
% in table form respectively. 
 i = 2;
[B,BNames] = randomEffects(fitglme(Design,model{1},'Distribution','binomial','FitMethod','Laplace'));

% The output of this is a 159*1 matrix and a table whose dimensions is
% 159*3. The output format contains three random effects parameters per
% subject (53*3 = 159) and arranged so that each subject has 3 values
% (intercept,reward,effort) before moving onto the next subject. 
% I will now create three vectors to represent our variables of interest. 

%Intrinsic Motivation or int represents the intercept variation per
%subject. index into all intercept values.  
intSen = B(1:4:end);
% reward sensitivity encoded as rewSen and indexes into all reward
% parameters
rewSen = B(2:4:end);
%effort sensitivity encoded as effSen 
effSen = B(3:4:end);

reweffSen = B(4:4:end);



% Now plot these for apathetic and non apathetic patients. 

% Intrinsic Motivation estimate

M = fitglme(Design,model{1},'Distribution','binomial','FitMethod','Laplace');

Fixed=M.fixedEffects;

% Create scores for each participant with both. 
Fixed_int = Fixed(1) + int;
Fixed_rew = Fixed(2) + rewSen;
Fixed_eff2 = Fixed(3) + effSen;
Fixed_reweff2 = Fixed(4) + reweffSen;

close all

apVec=apVec37;
subplot(2,2,1)

bar(1,fe(1)+(mean(int(apVec==0))),0.5);hold on;bar(2,fe(1)+(mean(int(apVec==1))),0.5);
errorbar(fe(1)+[(mean(int(apVec==0)))  (mean(int(apVec==1)))],...
    [std(int(apVec==0))/sqrt(length(int(apVec==0))) ...
    std(int(apVec==1))/sqrt(length(int(apVec==1)))],'k.','LineWidth',2);
ylabel('Parameter estimate')
xlabel('apathy status (no/yes)')
xticks([1 2]);
xticklabels({'noAp','Ap'})
title('Intrinsic Motivation')
set(gca,'fontSize',14,'fontWeight','bold')
 
 % reward Sensitivity parameter. 
 subplot(2,2,2)
 bar(1,fe(2)+(mean(rewSen(apVec==0))),0.5);hold on; bar(2,fe(2)+(mean(rewSen(apVec==1))),0.5);
 errorbar(fe(2)+[(mean(rewSen(apVec==0))) (mean(rewSen(apVec==1)))],...
     [std(rewSen(apVec==0))/sqrt(length(rewSen(apVec==0))) ...
     std(rewSen(apVec==1))/sqrt(length(rewSen(apVec==1)))],'k.','LineWidth',2);
 ylabel('Parameter estimate')
 xlabel('apathy status (no/yes)')
 xticks([1 2]);
 xticklabels({'noAp','Ap'})
 title('reward sensitivity')
 set(gca,'fontSize',14,'fontWeight','bold')
 
 
 % effort Sensitivity parameter. 
 subplot(2,2,3)
 bar(1,fe(3)+(mean(effSen(apVec==0))),0.5);hold on; bar(2,fe(3)+(mean(effSen(apVec==1))),0.5);
 errorbar(fe(3)+[(mean(effSen(apVec==0))) (mean(effSen(apVec==1)))],[std(effSen(apVec==0))/sqrt(length(effSen(apVec==0))) ...
     std(effSen(apVec==1))/sqrt(length(effSen(apVec==1)))],'k.','LineWidth',2);
 ylabel('Parameter estimate')
 xlabel('apathy status (no/yes)')
 xticks([1 2]);
 xticklabels({'noAp','Ap'})
 title('effort sensitivity')
 set(gca,'fontSize',14,'fontWeight','bold')
 
 subplot(2,2,4)
 bar(1,fe(3)+(mean(reweff(apVec==0))),0.5);hold on; bar(2,fe(3)+(mean(reweff(apVec==1))),0.5);
 errorbar(fe(3)+[(mean(reweff(apVec==0))) (mean(reweff(apVec==1)))],[std(reweff(apVec==0))/sqrt(length(reweff(apVec==0))) ...
     std(reweff(apVec==1))/sqrt(length(reweff(apVec==1)))],'k.','LineWidth',2);
 ylabel('Parameter estimate')
 xlabel('apathy status (no/yes)')
 xticks([1 2]);
 xticklabels({'noAp','Ap'})
 title('reweff')
 set(gca,'fontSize',14,'fontWeight','bold')


hold off 
hold off 
hold off


%% Can we now compute subjective value for every subject for every trial? 
% this is computed by V(r,e) = int + effSen + rewSen*rew
clear sub v i;
% fixed effects
zrew = (unique(rew))';
zeff = (unique(eff))';
fe=M.fixedEffects; % these are constant across subjects
for sub = 1:subj
    for r = 1:6
        for e = 1:6
           
            v(sub,r,e) = fe(1) + int(sub) ...
                + (fe(2) + rewSen(sub))*zrew(r)' ...
                + (fe(3) + effSen(sub))*(zeff(e))
            + ;
            
        end
    end
end
pp=1./(1 + exp(-v));



%%
clf
PLOT=3;
for i=1:subj
    subplot(10,9,i)
    if PLOT==1
        plot(squeeze(pp(i,:,:)));
        hold on
        set(gca,'colororderindex',1);
        plot(choices(:,:,i)','o:');
        hold off
    elseif PLOT==2
        imagesc(squeeze((pp(i,:,:))));
    elseif PLOT==3
        imagesc(squeeze(choices(:,:,i))');
    end
end

colormap(jet(256))
colorbar
% I now want to plot effort and reward curves with model fits overlayed. 



% plot the means of both models. 
% the structure of pp is (sub,r,e) whereas the structure of choices
% is(e,r,pp). first I need the means of both and then i need to make sure
% they are in the ssame dimension. 

% choices
close all
figure()
choicemean = mean(choices,3);
imagesc(flipud(choicemean));
colormap(jet(256));
colorbar
title('Choice map for all subjects','FontSize',16)
xlabel('Reward','FontSize',14);ylabel('Effort','FontSize',14);
yticklabels({'6' '5' '4' '3' '2' '1'});
set(gca,'FontSize',14,'FontWeight','bold');
hold on 
%probabilities
figure()
ppmean = squeeze(mean(pp));
imagesc(flipud(ppmean'));
title('Probability map for all subjects')
xlabel('Reward');ylabel('Effort');
yticklabels({'6' '5' '4' '3' '2' '1'});
set(gca,'FontSize',14,'FontWeight','bold');
colormap(jet(256));
colorbar



%% additional analysis related to computatoinal modelling
%% *********** RESULTS - EFFECTS OF REWARD AND EFFORT *************
% 3 way for groups all in one curve
close all
figure()
N = 6;
C = parula(N)
axes('ColorOrder',C,'NextPlot','replacechildren')
for i=1:6
    H1= errorBarPlot(squeeze(choices(i,:,apMedian==0))','-','LineWidth',5);hold on

end
legend('Motivated')

for i=1:6
    H2 = errorBarPlot(squeeze(choices(i,:,apMedian==1))',':','LineWidth',5);hold on
    ylim([0 1.1]);xlim([0 7]);
    title('Acceptance rates for each level of effort split by group'...
        ,'FontSize',24);
    ylabel('Acceptance Rate');
    xlabel('Reward (Apples)');
    ax=gca;
    set(ax,'fontWeight','bold','fontSize',18,'XTick',[1:1:6], ...
        'LineWidth',2,'XTickLabel',{'1','3','6','9','12','15'});
    
end
legend('NapE1','NapE2','NapE3','NapE4','NapE5','NapE6','ApE1','ApE2',...
    'ApE3','ApE4','ApE5','ApE6');


close all
figure()
EffLev = [1:6];

for i=1:6
    subplot(3,2,i)
%    H1= errorBarPlot(squeeze(choices(i,:,apMedian==0))','-','LineWidth',5);hold on
end
% set(gca,'ColorOrderIndex',1);
for i=1:6
    subplot(3,2,i)
%     H2 = errorBarPlot(squeeze(choices(i,:,apMedian==1))',':','LineWidth',5);hold on
    ylim([0 1.1]);xlim([0 7]);
    title(['Effort level' num2str(EffLev(i))],'FontSize',8);
    
    ylabel('Acceptance Rate');
    xlabel('Reward (Apples)');
    ax=gca;
    set(ax,'fontWeight','bold','fontSize',18,'XTick',[1:1:6], ...
        'LineWidth',2,'XTickLabel',{'1','3','6','9','12','15'});
    
end
legend('Not Apathetic','Apathetic');

    
% for each subject
    close
    for i = 1:6
        for s = 1:size(D.block,1)
            hold on
            subplot(12,12,s);
            plot(freqmap(i,:,s),'LineWidth',3);
            ylim([0 1.1]);xlim([0 7])
            ax=gca;
            set(ax,'fontWeight','bold','fontSize',18,'XTick',[1:1:6], ...
                'LineWidth',2,'XTickLabel',{'1','3','6','9','12','15'});
        end
    end
    legend('E1','E2','E3','E4','E5','E6')



% for subjects based on each sensitivity
% reward
close
clear s
N = 6;
C = hsv(N);
rere_minmedmax = [83 49 9 16 32 65 73 21 15];
for i = 1:6
    for s = 1:length(rere_minmedmax)
        hold on
        subplot(3,3,s);
        plot(freqmap(i,:,rere_minmedmax(s)),':','LineWidth',3,...
            'Color',C(i,:));
        ylim([0 1.1]);xlim([0 7])
        ax=gca;
        set(ax,'fontWeight','bold','fontSize',18,'XTick',[1:1:6], ...
            'LineWidth',2,'XTickLabel',{'1','3','6','9','12','15'});
    end
end
legend('E1','E2','E3','E4','E5','E6')

































































