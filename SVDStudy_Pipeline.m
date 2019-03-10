%% Script for Youssuf Saleh's Small Vessel Disease paper and thesis.  
%% Intention is for this to be the 'final' script for Small vessel disease 
% This includes data pre-processing, visualisation of data and analysis as
% well as the produciton of key MRI regressors. 
%% Performing relevant analyses including computational modelling. 
% 1. Converting excel sheet into a matlab table and extracting on excluded
% data points. 
% 2. Plotting a correlations table of all the questionnaires. 
% 3. Using a median split to look at 

D.block = D.block(:,37:216);



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
ylim([0 1.1]);xlim([0 7])
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:5], ...
    'XTickLabel',{'1','2','3','4','5','6'})
xlabel('Reward/Effort level')
ylabel('Prop. Offers Accepted')
ylim([0 1.1]); xlim([0.5 6.5]);xticks([1 2 3 4 5 6]);
title('SVD Group Performance across task');
hold off


[lgd, icons, plots, txt] = legend([H1.mainLine H2.mainLine],{'Reward','Effort'});


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
figure()
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



clear
close all

% Load Questionnaire data after exlusion

subj = size(D.R,1); %How many subjects?

% exclude RJ from SVD cohort. 

% create apathy vector (~debatable on exactly how to categorise patients)
apVec=[]';
for i = 1:subj
    if FQs_Ex.LARS_TOTAL(i)>-22 || FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.AES_TOTAL(i) > 37
    %if FQs_Ex.Composite(i) > nanmedian(FQs_Ex.Composite) + ...
            %nanstd(FQs_Ex.Composite)
        apVec(i)=1;
    else apVec(i)=0;
    end
end
    
apVec = apVec';
%apVec(9)=1; %aesVec for this subject 40
if 1 % if generating MRI inputs therefore need to exclude subj 2 and 19
    mriApVec = apVec;
    
end
    
% reshape choice data & prepare decision time matrix
choices = nanmean(grpD.choicemap{1},4);
forces = nanmean(grpD.maxforce{1},4);%all subjects average choices with accidental squeezes removed (nan)
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
scatterRegress(larsT,meanMVC);
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
 
for i=1:subj
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
for i=1:subj
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

b1=bar(1,mean(aaa(apVec==0,1)),'FaceColor',c('pastel blue'));hold on;  ...
errorbar(1,mean(aaa(apVec==0,1)),std(aaa(apVec==0,1))./sqrt(length(find(apVec==0))),'Color','k','LineWidth',3)
b2=bar(2,mean(aaa(apVec==1,1)),'FaceColor',c('pale red-violet'));hold on; 
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


% ****************  And plot correlation between lars_AI and acceptance *****************
figure();hold on
scatterRegress(larsSub56(:,3),accept/180,'MarkerEdgeColor',[0.5 0.5 0.5],'LineWidth',5)
ax=gca;
lsline(ax);
h = lsline(ax);
set(h,'LineWidth',3,'Color',[0.7 0.7 0.7])

ylabel('Proportion offers accepted')
xlabel('Action initiation subscale of LARS')
set(ax,'fontWeight','bold','FontSize',20,'YTick',0.2:0.2:1)
xlim([-4.5 1.5]);ylim([0.2 1.1])
axis square

if 0
figure()
errorBarPlot(cumAccept(apVec==0,:),'LineWidth',1);
hold on; errorBarPlot(cumAccept(apVec==1,:),'LineWidth',1);
xlim([0 200]); ylim ([0 200]);
title('cumulative offers accepted across experiment')
legend('No Apathy','Apathy')
end

%% *********** RESULTS - EFFECTS OF REWARD AND EFFORT *************
% Plot raw choice proportions in expanded form
close all
figure()
for i=1:6
    errorBarPlot(squeeze(choices(i,:,apVec==0))','--','LineWidth',5);hold on
end
set(gca,'ColorOrderIndex',1);
for i=1:6
    errorBarPlot(squeeze(choices(i,:,apVec==1))',':','LineWidth',5);hold on
end


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
    tempf(i,:)=reshape(forces(:,:,i)',36,1);
    tempflog(i,:)=log(temp(i,:));
end
    

%% **************** 2D plots ***********************
% using just errorbar function to avoid difficulties with errorBarPlot
c = @cmu.colors;

close all
subplot(1,3,1);
dat = squeeze(mean(choices,2))';
H1=shadedErrorBar(1:6,nanmean(dat(apVec==0,:)),   ...
    nanstd(dat(apVec==0,:)./sqrt(length(find(apVec==0)))),'lineprops',...
    {'color', c('air force blue')},...
    'patchSaturation',0.3);
hold on 
H2=shadedErrorBar(1:6,nanmean(dat(apVec==1,:)),   ...
    nanstd(dat(apVec==1,:)./sqrt(length(find(apVec==1)))),'lineprops',...
    {'color', c('brick red')},...
    'patchSaturation',0.3);axis square
ylim([0 1.1]);xlim([0.5 6.5])
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:6], ...
  'XTickLabel',{'1','2','3','4','5','6'})
xlabel('Reward/Effort level')
ylabel('Prop. Offers Accepted')
ylim([0.2 1])
title('Effort vs Apathy');
hold off


[lgd, icons, plots, txt] = legend([H1.mainLine H2.mainLine],{'No Apathy','Apathy'});


dat = squeeze(mean(choices,1))';
subplot(1,3,2)
H1=shadedErrorBar(1:6,nanmean(dat(apVec==0,:)),   ...
    nanstd(dat(apVec==0,:)./sqrt(length(find(apVec==0)))),'lineprops',...
    {'color', c('royal purple')},...
    'patchSaturation',0.3);
hold on 
H2=shadedErrorBar(1:6,nanmean(dat(apVec==1,:)),   ...
    nanstd(dat(apVec==1,:)./sqrt(length(find(apVec==1)))),'lineprops',...
    {'color', c('brick red')},...
    'patchSaturation',0.3);axis square
ylim([0 1.1]);xlim([0.5 6.5])
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:6], ...
  'XTickLabel',{'1','2','3','4','5','6'})
xlabel('Reward/Effort level')
ylabel('Prop. Offers Accepted')
ylim([0.2 1])
title('Reward vs Apathy');
hold off


[lgd, icons, plots, txt] = legend([H1.mainLine H2.mainLine],{'No Apathy','Apathy'});


%% Apathy - No Apathy plots (raw difference)
%  3D difference plot (2D plot not amazing...
subplot(1,3,3)
choiceDif=(mean(choices(:,:,apVec==0),3)-mean(choices(:,:,apVec==1),3));
h=surf(choiceDif);shading('interp');hold on;colormap('jet');%colorbar('Ticks',0:.05:.2)
ax=gca;
set(ax,'fontWeight','bold','fontSize',16,'XTick',[1:1:5],'YTickLabel',{'1','2','3','4','5'},'YTick',[1:1:5],'XTickLabel',{'1','2','3','4','5'},'ZTickLabel',{'','0','0.1','0.2','0.3','0.4','0.5'})
title('3D plot NoAp vs. AP')
ylabel('Effort (%MVC)')
xlabel('Reward')
zlabel('Diff. Proportion accepted')
hold on;
base=zeros(5,5);
hh=surf(base);
hh.FaceColor=[0.5 0.5 0.5];hh.FaceAlpha=1;
view(25,30)
if 1 % if want to add on grid lines
    for i=1:5
        plot3(1:5,(i)*ones(5,1),choiceDif(i,1:5),'k:','LineWidth',2)
        plot3((i)*ones(5,1),1:5,choiceDif(1:5,i),'k:','LineWidth',2)
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
close all

% check magnitude ofpeak squeeze at each force level
NAp_Sq = squeeze(nanmean(nanmean(grpD.maxforce{1}(:,:,apVec==0,:),4),2))';
Ap_Sq=   squeeze(nanmean(nanmean(grpD.maxforce{1}(:,:,apVec==1,:),4),2))';
%HC_Sq = squeeze(nanmean(nanmean(grpD_HC.maxforce{1},4),2))';
if 1 % limit max force exerted to 1
    NAp_Sq(NAp_Sq>1)=1;
    Ap_Sq(Ap_Sq>1)=1;
 %   HC_Sq(HC_Sq>1)=1;
end
figure()
%errorbar(nanmean(HC_Sq),nanstd(HC_Sq)./sqrt(19),':','Color',[0.7 0.7 0.7],'LineWidth',5);hold on
errorbar(nanmean(NAp_Sq),nanstd(NAp_Sq)./sqrt(19),'b:','LineWidth',5)
hold on
errorbar(nanmean(Ap_Sq),nanstd(Ap_Sq)./sqrt(11),'r:','LineWidth',5)
hold off
ax=gca;
ylim([0.2 1])
xlim([0 7])
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6],'XTickLabel',{0.1 0.24 0.38 0.52 0.66 0.80})
axis square
legend('  SVD - No Apathy','  SVD - Apathy')
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
ap=apVec;

close all
for i=1:length(find(ap==0))
    vecnoA(i) = 1+((rand-0.5)/5);
end
for i=1:length(find(ap==1))
    vecA(i) = 2+((rand-0.5)/5);
end
    
figure() % First just plot mean dt for the 3 groups

%bar(1,mean(nanmean(decisionTime_HC,2)),'FaceColor',[0.7 0.7 0.7]);hold on;
bar(1,mean(nanmean(DT(ap==0,:),2)),'FaceColor',color1);hold on
bar(2,mean(nanmean(DT(ap==1,:),2)),'FaceColor',color2);
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

%lets see what it looks like transformed. 
DT_transform = log(DT);
close
figure() % First just plot mean dt for the 3 groups

%bar(1,mean(nanmean(decisionTime_HC,2)),'FaceColor',[0.7 0.7 0.7]);hold on;
bar(1,nanmean(nanmean(DT_transform(ap==0,:),2)),'FaceColor',color1);hold on
bar(2,nanmean(nanmean(DT_transform(ap==1,:),2)),'FaceColor',color2);
if 1
 %   plot(vecHC,nanmean(decisionTime_HC,2),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5]);hold on
    plot(vecnoA,nanmean(DT_transform(ap==0,:),2),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5])
    plot(vecA,nanmean(DT_transform(ap==1,:),2),'.','MarkerSize',20,'MarkerEdgeColor',[0.5 0.5 0.5])
end
%errorbar(1,mean(nanmean(decisionTime_HC,2)),std(nanmean(decisionTime_HC,2))./sqrt(19),'k','LineWidth',3);
errorbar(1,nanmean(nanmean(DT_transform(ap==0,:),2)),std(nanmean(DT_transform(ap==0,:),2))./sqrt(length(find(ap==0))),'k','LineWidth',3);
errorbar(2,nanmean(nanmean(DT_transform(ap==1,:),2)),std(nanmean(DT_transform(ap==1,:),2))./sqrt(length(find(ap==1))),'k','LineWidth',3)
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',1:2,'XTickLabel',{'SVD-noAp','SVDAp'})
ylabel('Decision time (s)')
xlim([0.5 2.5])
%ylim([-5 5])
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
bar(1,nanmean(temp(ap==0)),'Facecolor',color1);hold on;errorbar(1,nanmean(temp(ap==0)),nanstd(temp(ap==0))./sqrt(length(find(ap==0))),'k','LineWidth',3);
bar(2,nanmean(temp(ap==1)),'Facecolor',color2);hold on;errorbar(2,nanmean(temp(ap==1)),nanstd(temp(ap==1))./sqrt(length(find(ap==1))),'k','LineWidth',3);
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
errorBarPlot(easyDec(ap==0,:),'Color',color1,'LineWidth',3);
hold on 
errorBarPlot(easyDec(ap==1,:),'Color',color2,'LineWidth',3);
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
figure();hold on
for i=1:5 % for each block
    errorBarPlot(squeeze(nanmean(grpD.choicemap{1}(:,:,apVec==1,i),1))','LineWidth',3);
end
legend('Block 1','Block 2','Block 3','Block 4','Block 5');
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6])
ylim([0.2 1.1]);xlim([0 7]);
%figure();hold on
%for i=1:5 % for each block
 %   errorBarPlot(squeeze(nanmean(grpD_HC.choicemap{1}(:,:,:,i),1))','LineWidth',3);
%end
legend('Block 1','Block 2','Block 3','Block 4','Block 5');
ax=gca;
set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:6])
ylim([0.2 1.1]);xlim([0 7]);

%% ******************** Depression Effects   *************************
% Dysphoria subscale does not really split apathetic group up (using median
% split) - can argue anyway to use GDS given prevalence in SVD cohorts... 
close all

depVec=[];
for i=1:subj
    if FQs_Ex.BDI(i)<=13
        depVec(i)=0;
    else
        depVec(i)=1;
    end
end
depVec=depVec';


for i=1:2
    figure();
    dat = squeeze(mean(choices,i))';
   % dat_hc=squeeze(mean(choices_HC,i))';
    %errorbar(1:6,mean(dat_hc),std(dat_hc)./sqrt(19),'m--','LineWidth',3); hold on;
    errorbar(1:6,mean(dat(depVec==0,:)),std(dat(depVec==0,:))./sqrt(length(depVec==0)),'b','LineWidth',3); hold on;
    errorbar(1:6,mean(dat(depVec==1,:)),std(dat(depVec==1,:))./sqrt(length(depVec==1)),'r','LineWidth',3)
    %title('proportion of offers accepted as reward level increases')
    legend('SVDc No Depr','SVDc Depr');
    axis square
    ylim([0 1.1]);xlim([0 7])
    ax=gca;
    if i==1
        set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:1:6],'XTickLabel',{'10','24','38','52','66','80'})
        xlabel('Effort level (% MVC)')
        ylabel('Proportion of offers accepted')
    else
        set(ax,'fontWeight','bold','fontSize',20,'XTick',[1:1:6],'XTickLabel',{'1','3','6','9','12','15'})
        xlabel('Reward level')
        ylabel('Proportion of offers accepted')
    end
end
%% Dysphoria subscale
%load bdi_full_cad
bdi_full=xxx;
qq=[1 2 3 5 6 7 19 9 10 11 14]; % dysphoria subscale

BDI_dys=[];
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
% explain the near significant result that we have. 
for i = 1:subj
  for t = 1:180
    if D.stake(i,t) < 9 % low reward scores a 1. High reward scores 0. 
      
      D.catRew(i,t)=1;
    else D.catRew(i,t)= 2;
    end
  end
end


%zscores for qs.
% AES and LARS z scored
Z_AES = nanzscore(FQs_Ex.AES_TOTAL);
Z_LARSt = nanzscore(FQs_Ex.LARS_AI);
Z_Dep = nanzscore(FQs_Ex.BDI);
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
    catRew = D.catRew(i,:)';
    block = D.block(i,:)';
    % create a matrix of decision times that will be excluded from
    % the final decision matrix that corresponds to those DT which
    % are more than 3 STD from the mean for that particular patient
    dtind = dt(i,:)';
    RT_Slow = abs(dtind - nanmean(dtind)) <= 3.*nanstd(dtind);
    a=1;
    for j = 1:length(decVec) %for each of the 180 trials
        if decVec(j) < 0.4
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
        catRew(removal)    = [];
        block(removal)     = [];
    end
    clear removal
    if ~linear
        subData = [subData;i*ones(length(choicesVec),1) choicesVec, ...
            vigVec decVec Z_decVec reward effort                    ...
            apVec(i)*ones(length(choicesVec),1),                    ...
            Z_AES(i)*ones(length(choicesVec),1),                    ...
            Z_LARSt(i)*ones(length(choicesVec),1),                  ...
            Composite(i)*ones(length(choicesVec),1),                ...
            Z_Dep(i)*ones(length(choicesVec),1),                    ...
            RT_Slow(i)*ones(length(choicesVec),1), ...
            Z_AES_E(i)*ones(length(choicesVec),1), ...
            catRew, block];
    end
end

subject   = categorical(subData(:,1));
choice    = subData(:,2);
Vigour    = subData(:,3);
DT        = subData(:,4);
Z_DT      = subData(:,5);
rew       = subData(:,6);
eff       = subData(:,4);
ap        = subData(:,8);
AES_T     = subData(:,9);
LARSt_Z   = subData(:,10);
CompAp    = subData(:,11);
Depression= subData(:,12);
RT_slow   = subData(:,13);
AES_E     = subData(:,14);
catRew    = categorical(subData(:,15));
Block     = categorical(subData(:,16));

if 0
    % ****** IF WANT Quadratic effort *******
    eff = eff.^2;
end
if 1
    rew=nanzscore(rew);
    eff=nanzscore(eff);
    ap = nanzscore(ap);
    catRew = nanzscore(catRew);
    Block = nanzscore(Block);
end
    %force=nanzscore(force);
    %lars = zscore(lars);

    Design = table(choice,rew,eff,ap,AES_T,AES_E,LARSt_Z,CompAp,Depression, ...
      subject,catRew,Block);


%%
clear aic glme_fit bicf
linear=0; % which model type to run
models = {    
    
%  First model will look at composite apathy score  and depression as ...
%continuous variables with 4 way interactions and apathy*depression
%interactions taken out. 
'choice ~ rew*eff*CompAp  + rew*eff*Depression +  (1|subject)'
% Then just the AES as a continuous variable. 
'choice ~ rew*eff*AES_T   + rew*eff*Depression +  (1|subject)'
% include block 
'choice ~ rew*eff*AES_T*Block   + rew*eff*Depression*Block +  (1|subject)'
% Then the lars Total
'choice ~ rew*eff*LARSt_Z + rew*eff*Depression +  (1|subject)'
%use the apathy cut off (median split)
'choice ~ rew*eff*ap + rew*eff*Depression +  (1|subject)'
% full 4 model factorial using the same 4 apathy variables ... 
%just to compare model fits and effects. 
%'choice ~ rew*eff*CompAp*Depression +  (1|subject)'
%'choice ~ rew*eff*AES_T*Depression +  (1|subject)'
%'choice ~ rew*eff*LARSt_Z*Depression +  (1|subject)'
%'choice ~ rew*eff*ap*Depression +  (1|subject)'

% Now look at models that do not include depression to compare model fits. 
'choice ~ rew*eff*CompAp  +  (1|subject)'
'choice ~ rew*eff*AES_T  +  (1|subject)'
'choice ~ rew*eff*AES_T*Block  +  (1|subject)'
'choice ~ rew*eff*ap +  (1|subject)'
% What about one that includes depression only
'choice ~ rew*eff*Depression + (1|subject)'
% model which looks at the different levels of reward 
'choice ~ catRew*AES_T + (1|subject)'
    };

if linear
    for i=1:length(models)
        sprintf('starting model %g',i)
        glme_fit{i}=fitglme(Design,models{i},'Distribution','normal');
        aic(i)=glme_fit{i}.ModelCriterion.AIC;
        bicf(i)=glme_fit{i}.ModelCriterion.BIC;
    end
else
    for i=1:length(models)
        sprintf('starting model %g',i)
        glme_fit{i}=fitglme(Design,models{i},'Distribution','binomial','fitmethod','Laplace','PLIterations',1000000);
        aic(i)=glme_fit{i}.ModelCriterion.AIC;
        bicf(i)=glme_fit{i}.ModelCriterion.BIC;
                

    end
end
aic=aic-min(aic);
bicf=bicf-min(bicf);

if 0
    save('glme_fit_lin_cat','models','glme_fit')
end
if 0 % save outputs
save('glme_fitPD_Z','models','glme_fit')
end


%% Analysis for Force. 

% first transform

force  = nanzscore(log(force));
% check that data is normalised 
hist(force)
%create Table with normalised data. 
Design1 = table(force,rew,eff,ap,subject);
%assign linearity
linear = 1;

% create models
models_force = {
 
'force ~ rew*eff*ap + (1|subject)'
'force ~ rew*eff*ap + (1+rew| subject)' 
'force ~ rew*eff*ap + (1+rew+eff| subject)' 
'force ~ rew*eff + rew*ap + eff*ap + (1|subject)'
'force ~ rew*eff + (1|subject)'
  
  };

  
if linear
    for i=1:length(models_force)
        sprintf('starting model %g',i)
        glme_fit_force{i}=fitglme(Design1,models_force{i},'Distribution','normal');
        aicf(i)=glme_fit_force{i}.ModelCriterion.AIC;
        bicf(i)=glme_fit_force{i}.ModelCriterion.BIC;
    end
else
    for i=1:length(models_force)
        sprintf('starting model %g',i)
        glme_fit_force{i}=fitglme(Design1,models_force{i},'Distribution','binomial','fitmethod','Laplace');
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



%% Abbreviated version of Computational modelling.
% First we To calculate the values for Intrinsic motivation, reward ...
%sensitivity and effort sensitivity fir each patient. 

% First use the appropriate glmemodel
model = { ...
    'choice ~ 1 + rew + eff + (rew+eff|subject)'
    };

% Then run the GLME with the two main outputs being B and BNames which
% correspond to the random effects parameters and their corresponding names
% in table form respectively. 
i = 1;
[B,BNames] = randomEffects(fitglme(Design,model{i},'Distribution','binomial','fitmethod','Laplace'));

% The output of this is a 159*1 matrix and a table whose dimensions is
% 159*3. The output format contains three random effects parameters per
% subject (53*3 = 159) and arranged so that each subject has 3 values
% (intercept,reward,effort) before moving onto the next subject. 
% I will now create three vectors to represent our variables of interest. 

%Intrinsic Motivation or int represents the intercept variation per
%subject. index into all intercept values.  
int = B(1:3:end);
% reward sensitivity encoded as rewSen and indexes into all reward
% parameters
rewSen = B(2:3:end);
%effort sensitivity encoded as effSen 
effSen = B(3:3:end);


% Now plot these for apathetic and non apathetic patients. 

% Intrinsic Motivation estimate
M = fitglme(Design,model{i},'Distribution','binomial','fitmethod','Laplace');
fe=M.fixedEffects;
close all


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
                + (fe(3) + effSen(sub))*(zeff(e));
            
        end
    end
end
pp=1./(1 + exp(-v));



%%
clf
PLOT=2;
for i=1:subj
    subplot(7,8,i)
    if PLOT==1
        plot(squeeze(pp(i,:,:)));
        hold on
        set(gca,'colororderindex',1);
        plot(choices(:,:,i)','o:');
        hold off
    elseif PLOT==2
        imagesc((pp(i,:,:)));
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












































































