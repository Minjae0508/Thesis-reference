classdef Image_Processing_App_for_Thesis_Reference_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        SaveasButton                    matlab.ui.control.Button
        DFTSwitchLabel_2                matlab.ui.control.Label
        GrayscaleSwitch                 matlab.ui.control.Switch
        GrayscaleSwitchLabel            matlab.ui.control.Label
        gamma1Button_5                  matlab.ui.control.Button
        gamma1Button_4                  matlab.ui.control.Button
        gamma1Button_3                  matlab.ui.control.Button
        gamma1Button_2                  matlab.ui.control.Button
        tgv                             matlab.ui.control.NumericEditField
        gamma1Button                    matlab.ui.control.Button
        HistogramLabel                  matlab.ui.control.Label
        AllResetButton                  matlab.ui.control.Button
        TabGroup                        matlab.ui.container.TabGroup
        DRMTab                          matlab.ui.container.Tab
        krEELabel                       matlab.ui.control.Label
        ResetButton_4                   matlab.ui.control.Button
        GoButton_4                      matlab.ui.control.Button
        Cst                             matlab.ui.control.NumericEditField
        ContrastStretchingTransformationLabel  matlab.ui.control.Label
        ResetButton_3                   matlab.ui.control.Button
        GoButton_3                      matlab.ui.control.Button
        Ls                              matlab.ui.control.NumericEditField
        gcxlogfLabel                    matlab.ui.control.Label
        LogarithmicTransformationLabel  matlab.ui.control.Label
        DynamicRangeManipulationLabel   matlab.ui.control.Label
        MaskingTab                      matlab.ui.container.Tab
        Label_2                         matlab.ui.control.Label
        abcaxbandxcisdefaultvalueLabel  matlab.ui.control.Label
        Usfr                            matlab.ui.control.NumericEditField
        UnsharpFilterRaiusLabel         matlab.ui.control.Label
        Usfa                            matlab.ui.control.NumericEditField
        AmountLabel                     matlab.ui.control.Label
        ResetButton_9                   matlab.ui.control.Button
        GoButton_9                      matlab.ui.control.Button
        Label                           matlab.ui.control.Label
        Lfa                             matlab.ui.control.NumericEditField
        LaplacianFilteralphaLabel       matlab.ui.control.Label
        ResetButton_8                   matlab.ui.control.Button
        GoButton_8                      matlab.ui.control.Button
        SobelFilterSwitch               matlab.ui.control.Switch
        SobelFilterSwitchLabel          matlab.ui.control.Label
        PrewittFliterSwitch             matlab.ui.control.Switch
        PrewittFliterSwitchLabel        matlab.ui.control.Label
        LoGfs1                          matlab.ui.control.NumericEditField
        LoGFiltersizeLabel              matlab.ui.control.Label
        LoGfg                           matlab.ui.control.NumericEditField
        sigmaLabel_2                    matlab.ui.control.Label
        ResetButton_7                   matlab.ui.control.Button
        GoButton_7                      matlab.ui.control.Button
        LoGfs2                          matlab.ui.control.NumericEditField
        XLabel_3                        matlab.ui.control.Label
        Mfl                             matlab.ui.control.NumericEditField
        MotionFilterLinearLabel         matlab.ui.control.Label
        Mft                             matlab.ui.control.NumericEditField
        ThetaLabel                      matlab.ui.control.Label
        ResetButton_6                   matlab.ui.control.Button
        GoButton_6                      matlab.ui.control.Button
        Dfr                             matlab.ui.control.NumericEditField
        DiskFilterRaidussizeLabel       matlab.ui.control.Label
        ResetButton_5                   matlab.ui.control.Button
        GoButton_5                      matlab.ui.control.Button
        SpatialFilteringLabel           matlab.ui.control.Label
        Gfg                             matlab.ui.control.NumericEditField
        sigmaLabel                      matlab.ui.control.Label
        ResetButton_2                   matlab.ui.control.Button
        GoButton_2                      matlab.ui.control.Button
        Gfs2                            matlab.ui.control.NumericEditField
        XLabel_2                        matlab.ui.control.Label
        Gfs1                            matlab.ui.control.NumericEditField
        GaussianFiltersizeLabel         matlab.ui.control.Label
        ResetButton                     matlab.ui.control.Button
        GoButton                        matlab.ui.control.Button
        Afs2                            matlab.ui.control.NumericEditField
        XLabel                          matlab.ui.control.Label
        Afs1                            matlab.ui.control.NumericEditField
        AverageFiltersizeLabel          matlab.ui.control.Label
        PassfilterTab                   matlab.ui.container.Tab
        Label_4                         matlab.ui.control.Label
        Label_3                         matlab.ui.control.Label
        Nfc2                            matlab.ui.control.NumericEditField
        Nfr2                            matlab.ui.control.NumericEditField
        ResetButton_17                  matlab.ui.control.Button
        GoButton_17                     matlab.ui.control.Button
        Nfc1                            matlab.ui.control.NumericEditField
        NotchfilteringcolumnLabel       matlab.ui.control.Label
        Nfr1                            matlab.ui.control.NumericEditField
        NotchfilteringrowLabel          matlab.ui.control.Label
        ResetButton_16                  matlab.ui.control.Button
        GoButton_16                     matlab.ui.control.Button
        widthLabel_2                    matlab.ui.control.Label
        Ibrw                            matlab.ui.control.NumericEditField
        Ibrr                            matlab.ui.control.NumericEditField
        IdealBandrejectfilterBandradiusLabel  matlab.ui.control.Label
        ResetButton_15                  matlab.ui.control.Button
        GoButton_15                     matlab.ui.control.Button
        widthLabel                      matlab.ui.control.Label
        Ibpw                            matlab.ui.control.NumericEditField
        Ibpr                            matlab.ui.control.NumericEditField
        IdealBandpassfilterBandradiusLabel  matlab.ui.control.Label
        ResetButton_14                  matlab.ui.control.Button
        GoButton_14                     matlab.ui.control.Button
        orderLabel_2                    matlab.ui.control.Label
        Bwlo                            matlab.ui.control.NumericEditField
        Bwlf                            matlab.ui.control.NumericEditField
        ButterworthLowPassfilterCutofffrequencyLabel  matlab.ui.control.Label
        ResetButton_13                  matlab.ui.control.Button
        GoButton_13                     matlab.ui.control.Button
        Bwho                            matlab.ui.control.NumericEditField
        orderLabel                      matlab.ui.control.Label
        Bwhf                            matlab.ui.control.NumericEditField
        ButterworthHighPassfilterCutofffrequencyLabel  matlab.ui.control.Label
        ResetButton_12                  matlab.ui.control.Button
        GoButton_12                     matlab.ui.control.Button
        Ilp                             matlab.ui.control.NumericEditField
        IdealLowPassfilterCutofffrequencyLabel  matlab.ui.control.Label
        ResetButton_11                  matlab.ui.control.Button
        GoButton_11                     matlab.ui.control.Button
        Ihp                             matlab.ui.control.NumericEditField
        IdealHighPassfilterCutofffrequencyLabel  matlab.ui.control.Label
        ResetButton_10                  matlab.ui.control.Button
        GoButton_10                     matlab.ui.control.Button
        FrequencyDomainLabel            matlab.ui.control.Label
        NoiseTab                        matlab.ui.container.Tab
        SpecklenoiseSwitch              matlab.ui.control.Switch
        SpecklenoiseSwitchLabel         matlab.ui.control.Label
        SaltPeppernoiseSwitch           matlab.ui.control.Switch
        SaltPeppernoiseSwitchLabel      matlab.ui.control.Label
        PoissonnoiseSwitch              matlab.ui.control.Switch
        PoissonnoiseSwitchLabel         matlab.ui.control.Label
        GaussiannoiseSwitch             matlab.ui.control.Switch
        GaussiannoiseSwitchLabel        matlab.ui.control.Label
        RestorationTab                  matlab.ui.container.Tab
        iteration                       matlab.ui.control.NumericEditField
        IterativeNonLinearRestorationIterationSwitch  matlab.ui.control.Switch
        IterativeNonLinearRestorationIterationSwitchLabel  matlab.ui.control.Label
        WeinerfilterSwitch              matlab.ui.control.Switch
        WeinerfilterSwitchLabel         matlab.ui.control.Label
        mnf                             matlab.ui.control.NumericEditField
        Minfiltersize3Label             matlab.ui.control.Label
        ResetButton_23                  matlab.ui.control.Button
        GoButton_23                     matlab.ui.control.Button
        mxf                             matlab.ui.control.NumericEditField
        Maxfiltersize3Label             matlab.ui.control.Label
        ResetButton_22                  matlab.ui.control.Button
        GoButton_22                     matlab.ui.control.Button
        mfm                             matlab.ui.control.NumericEditField
        MedianfiltermxnLabel            matlab.ui.control.Label
        ResetButton_21                  matlab.ui.control.Button
        GoButton_21                     matlab.ui.control.Button
        mfn                             matlab.ui.control.NumericEditField
        XLabel_7                        matlab.ui.control.Label
        hmm                             matlab.ui.control.NumericEditField
        HarmonicmeanmxnLabel            matlab.ui.control.Label
        ResetButton_19                  matlab.ui.control.Button
        GoButton_19                     matlab.ui.control.Button
        hmn                             matlab.ui.control.NumericEditField
        XLabel_5                        matlab.ui.control.Label
        armm                            matlab.ui.control.NumericEditField
        ArithmeticmeanmxnLabel          matlab.ui.control.Label
        ResetButton_18                  matlab.ui.control.Button
        GoButton_18                     matlab.ui.control.Button
        armn                            matlab.ui.control.NumericEditField
        XLabel_4                        matlab.ui.control.Label
        ReconstructionTab               matlab.ui.container.Tab
        FilterDropDown                  matlab.ui.control.DropDown
        FilterDropDownLabel             matlab.ui.control.Label
        InterpolationDropDown           matlab.ui.control.DropDown
        InterpolationDropDownLabel      matlab.ui.control.Label
        BackProjectiontheta0EditFieldLabel  matlab.ui.control.Label
        irtm                            matlab.ui.control.NumericEditField
        Projectiontheta0EditFieldLabel  matlab.ui.control.Label
        rtm                             matlab.ui.control.NumericEditField
        RadonTransformLabel             matlab.ui.control.Label
        ProjectionthetaEditFieldLabel_2  matlab.ui.control.Label
        rtn                             matlab.ui.control.NumericEditField
        ProjectionthetaEditFieldLabel_3  matlab.ui.control.Label
        irtn                            matlab.ui.control.NumericEditField
        GoButton_24                     matlab.ui.control.Button
        ResetButton_24                  matlab.ui.control.Button
        RT                              matlab.ui.control.UIAxes
        AffinetransformTab              matlab.ui.container.Tab
        ResetButton_25                  matlab.ui.control.Button
        ReflectionLabel                 matlab.ui.control.Label
        GoButton_30                     matlab.ui.control.Button
        dx                              matlab.ui.control.NumericEditField
        TranslationdxdyLabel            matlab.ui.control.Label
        GoButton_29                     matlab.ui.control.Button
        dy                              matlab.ui.control.NumericEditField
        Label_6                         matlab.ui.control.Label
        shearv                          matlab.ui.control.NumericEditField
        ShearverticalslopeLabel         matlab.ui.control.Label
        GoButton_28                     matlab.ui.control.Button
        shearh                          matlab.ui.control.NumericEditField
        ShearHorizontalslopeLabel       matlab.ui.control.Label
        GoButton_27                     matlab.ui.control.Button
        rotation                        matlab.ui.control.NumericEditField
        RotaionthetaLabel               matlab.ui.control.Label
        GoButton_26                     matlab.ui.control.Button
        scalingx                        matlab.ui.control.NumericEditField
        ScalingSxSyLabel                matlab.ui.control.Label
        GoButton_25                     matlab.ui.control.Button
        scalingy                        matlab.ui.control.NumericEditField
        Label_5                         matlab.ui.control.Label
        AHETab                          matlab.ui.container.Tab
        ahe                             matlab.ui.control.Switch
        DistributionDropDown            matlab.ui.control.DropDown
        DistributionDropDownLabel       matlab.ui.control.Label
        RangeDropDown                   matlab.ui.control.DropDown
        RangeDropDownLabel              matlab.ui.control.Label
        Default001Label                 matlab.ui.control.Label
        cl                              matlab.ui.control.NumericEditField
        ClipLimitLabel                  matlab.ui.control.Label
        AdapticeHistogramEqualizationLabel  matlab.ui.control.Label
        IntensityGammaLabel             matlab.ui.control.Label
        ResultImageLabel                matlab.ui.control.Label
        EqualizationSwitch              matlab.ui.control.Switch
        EqualizationSwitchLabel         matlab.ui.control.Label
        GammaSlider                     matlab.ui.control.Slider
        OriginalImageLabel              matlab.ui.control.Label
        ImageinformationButton          matlab.ui.control.Button
        SelectimagefileButton           matlab.ui.control.Button
        DFTAxes                         matlab.ui.control.UIAxes
        ImageHist                       matlab.ui.control.UIAxes
        RedAxes                         matlab.ui.control.UIAxes
        GreenAxes                       matlab.ui.control.UIAxes
        BlueAxes                        matlab.ui.control.UIAxes
        RI                              matlab.ui.control.UIAxes
        ImageAxes                       matlab.ui.control.UIAxes
    end

    methods (Access = private)
        
        
        
        function updateimage(app,imagefile)
            % For corn.tif, read the second image in the file 
            global im h Bpw Brw

            if strcmp(imagefile,'corn.tif')
                im = imread('corn.tif');
            else
                try
                    im = imread(imagefile);
                catch 
                    return;
                end            
            end 
            Bpw=0;
            Brw=0;
            
            % Create histograms based on number of color channels
            switch size(im,3)
                case 1
                    % Display the grayscale image
                    imagesc(app.ImageAxes,im);
                    imagesc(app.RI,im);
                    
                    % Plot all histograms with the same data for grayscale
                    histr = histogram(app.RedAxes, im, 'FaceColor',[1 0 0],'EdgeColor', 'none');
                    histg = histogram(app.GreenAxes, im, 'FaceColor',[0 1 0],'EdgeColor', 'none');
                    histb = histogram(app.BlueAxes, im, 'FaceColor',[0 0 1],'EdgeColor', 'none');
                    
                case 3
                    % Display the truecolor image
                    imagesc(app.ImageAxes,im);
                    imagesc(app.RI,im);
                    % Plot the histograms
                    histr = histogram(app.RedAxes, im(:,:,1), 'FaceColor', [1 0 0], 'EdgeColor', 'none');
                    histg = histogram(app.GreenAxes, im(:,:,2), 'FaceColor', [0 1 0], 'EdgeColor', 'none');
                    histb = histogram(app.BlueAxes, im(:,:,3), 'FaceColor', [0 0 1], 'EdgeColor', 'none');
                    
                otherwise
                    % Error when image is not grayscale or truecolor
                    uialert(app.UIFigure, 'Image must be grayscale or truecolor.', 'Image Error');
                    return;
            end
                % Get largest bin count
                maxr = max(histr.BinCounts);
                maxg = max(histg.BinCounts);
                maxb = max(histb.BinCounts);
                maxcount = max([maxr maxg maxb]);
                
                % Set y axes limits based on largest bin count
                app.RedAxes.YLim = [0 maxcount];
                app.RedAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
                app.GreenAxes.YLim = [0 maxcount];
                app.GreenAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
                app.BlueAxes.YLim = [0 maxcount];
                app.BlueAxes.YTick = round([0 maxcount/2 maxcount], 2, 'significant');
                h=im;
 
        end
        
        function    fhat = amean(g,mn)
                    w=fspecial('average',mn);
                    fhat=imfilter(g,w,'replicate');
        end
        
        function  fhat=hmean(g,m,n)
                   g2=im2double(g);
                  fhat=m*n./imfilter(1./(g2),ones(m,n),'replicate');
        end
        

 

        
        
        function image(app,imagefile)
            
            global  im h Cst ls Afs1 Afs2 Dfr Gfs1 Gfs2 Gfg LoGfs1 LoGfs2 LoGfg Mft Mfl Lfa Usfa Usfr Ihp Ilp Bwhf Bwfo Bwlo  Bwlf Ibpr Ibpw Ibrw Ibrr Nfr1 Nfr2 Nfc1 Nfc2
            global  armm armn hmm hmn mfm mfn mxf mnf 
            global rtm rtn irtm irtn rdff rdfi
            global hn hf PSF np iteration

            gamma=app.GammaSlider.Value;
            equal=app.EqualizationSwitch.Value;
            ahe=app.ahe.Value;
            gray=app.GrayscaleSwitch.Value;
            prewitt=app.PrewittFliterSwitch.Value;
            Sobel=app.SobelFilterSwitch.Value;
            gn=app.GaussiannoiseSwitch.Value;
            pn=app.PoissonnoiseSwitch.Value;
            sn=app.SpecklenoiseSwitch.Value;
            spn=app.SaltPeppernoiseSwitch.Value;
            wf=app.WeinerfilterSwitch.Value;
            
            Lfa=app.Lfa.Value;
            
           
           % Equalization
            if strcmp(equal,'On')
            h=histeq(imagefile);
            else
            h=imagefile;
            end
            
            % Equalization
            if strcmp(ahe,'On')
            h=adapthisteq(h,'ClipLimit',app.cl.Value,'Range',app.RangeDropDown.Value,'Distribution',app.DistributionDropDown.Value);
            else
            end
            
           % Grayscale
            if strcmp(gray,'On')
                h=im2gray(h);
            else
            end
           
           % Prewitt
           if strcmp(prewitt,'On')
               g=fspecial('prewitt');
               h=imfilter(h,g);
               PSF=g;
           else
           end
           
           % Sobel
           if strcmp(Sobel,'On')
               g=fspecial('sobel');
               h=imfilter(h,g);
               PSF=g;
           else
           end
            
           % Intensity gamma
            h=imadjust(h,[],[],gamma); 
            
           % Logarithmic transformation
            if ls>0
            h=double(h);    
            h=log(1+h); 
            h=mat2gray(h);
            else
            end
            
           % Contrast-Stretching transformation
            if Cst>0.1
            h=double(h);    
            k=mean2(h);
            h=1./(1+(k./h).^Cst);
            h=mat2gray(h);
            else
            end  

           % Averiging filter
            if Afs1>0
            g=fspecial('average',[Afs1,Afs2]);
            h=imfilter(h,g);
            PSF=g;
            else
            end
            
           % Disk filter
            if Dfr>0
           g=fspecial('Disk',Dfr);
           h=imfilter(h,g);
           PSF=g;
           else
           end    
           

           % Gaussian filter
            if Gfs1>0
            g=fspecial('Gaussian',[Gfs1,Gfs2],Gfg);
            h=imfilter(h,g);
            PSF=g;
            else
            end
            
           % Laplacian filter
            if Lfa>0.0000001
            g=fspecial('laplacian',Lfa);
            h=imfilter(h,g);
            PSF=g;
            else
            end 
            
           % LoG filter
            if LoGfg>0 
            g=fspecial('log',[LoGfs1, LoGfs2],LoGfg);
            h=imfilter(h,g); 
            PSF=g;
            else
            end
            
           % Motionfilter
            if Mft>0.1
            g=fspecial('motion',Mfl,Mft);
            h=imfilter(h,g);
            PSF=g;
            else
            end
            
           % Unsharp filter
            if Usfr>0.1
            h=imsharpen(h,'Radius',Usfr,'Amount',Usfa);
            PSF=g;
            else
            end
            
              
           % Ideal High Pass filter
            if Ihp>0
                if strcmp(gray,'On')
            [M, N]=size(h);
            FT_img = fft2(double(h));
            D0 = Ihp;  
            u = 0:(M-1);
            idx = find(u>M/2);
            u(idx) = u(idx)-M;
            v = 0:(N-1);
            idy = find(v>N/2);
            v(idy) = v(idy)-N;  
            [V, U] = meshgrid(v, u);  
            D = sqrt(U.^2+V.^2);
            H = double(D > D0);
            G = H.*FT_img;
            h = real(ifft2(double(G)));
                else
            app.GrayscaleSwitch.Value='On';
            image(app,im)
                end
            else
            end
            
            
           % Ideal Low Pass filter
            if Ilp>0
                if strcmp(gray,'On')
                    
            [M, N] = size(h);
            FT_img = fft2(double(h));
            D0 = Ilp;
            u = 0:(M-1);
            idx = find(u>M/2);
            u(idx) = u(idx)-M;
            v = 0:(N-1);
            idy = find(v>N/2);
            v(idy) = v(idy)-N;
            [V, U] = meshgrid(v, u);
            D = sqrt(U.^2+V.^2);
            H = double(D <= D0);
            G = H.*FT_img;
            h = real(ifft2(double(G)));
            else
            app.GrayscaleSwitch.Value='On';
            image(app,im)
                end
            else
            end
            
            
          % Butterwoth High Pass filter   
             if Bwhf>0.1
                if strcmp(gray,'On')
            [M, N] = size(h);
            FT_img = fft2(double(h));
            n = Bwfo;
            D = Bwhf;
            u = 0:(M-1);
            v = 0:(N-1);
            idx = find(u > M/2);
            u(idx) = u(idx) - M;
            idy = find(v > N/2);
            v(idy) = v(idy) - N;
            [V, U] = meshgrid(v, u);
            D0 = sqrt(U.^2 + V.^2);
            H = 1./(1 + (D./D0).^(2*n));
            G = H.*FT_img;
            h = real(ifft2(double(G)));
                else
                end
            end
            
            % Butterwoth Low Pass filter   
            if Bwlf>0.1
                if strcmp(gray,'On')
            [M, N] = size(h);
            FT_img = fft2(double(h));
            n = Bwlo;
            D0 = Bwlf;
            u = 0:(M-1);
            v = 0:(N-1);
            idx = find(u > M/2);
            u(idx) = u(idx) - M;
            idy = find(v > N/2);
            v(idy) = v(idy) - N;
            [V, U] = meshgrid(v, u);
            D = sqrt(U.^2 + V.^2);
            H = 1./(1 + (D./D0).^(2*n));
            G = H.*FT_img;
            h = real(ifft2(double(G)));
                else
                end
            end
            
            % Ideal Bandpass filter
            if Ibpw>0
                if strcmp(gray,'On')
            [M, N]=size(h);
            FT_img = fft2(double(h));
            w=Ibpw;
            r=Ibpr;
            D1=r-w/2;
            D2=r+w/2;
            u = 0:(M-1);
            idx = find(u>M/2);
            u(idx) = u(idx)-M;
            v = 0:(N-1);
            idy = find(v>N/2);
            v(idy) = v(idy)-N;
            [V, U] = meshgrid(v, u);
            D = sqrt(U.^2+V.^2);
            H1 = double(D > D1);
            H2 = double(D < D2);
            G = H2.*H1.*FT_img;
            h = real(ifft2(double(G)));
            else
            app.GrayscaleSwitch.Value='On';
            image(app,im)
                end
            else
            end
            
            % Ideal Bandreject filter
            if Ibrw>0
               if strcmp(gray,'On')
            [M, N]=size(h);
            FT_img = fft2(double(h));
            w=Ibrw;
            r=Ibrr;
            D1=r-w/2;
            D2=r+w/2;
            u = 0:(M-1);
            idx = find(u>M/2);
            u(idx) = u(idx)-M;
            v = 0:(N-1);
            idy = find(v>N/2);
            v(idy) = v(idy)-N;
            [V, U] = meshgrid(v, u);
            D = sqrt(U.^2+V.^2);
            H1 = (double(D > D1)-1)*(-1);
            H2 = (double(D < D2)-1)*(-1);
            G = H2.*H1.*FT_img;
            h = real(ifft2(double(G)));
            else
            app.GrayscaleSwitch.Value='On';
            image(app,im)
                end
            else
            end
            
            % Notch filter row
            if Nfr1>0
               if strcmp(gray,'On') 
              F=fft2(double(h));
              F(Nfr1:1:Nfr2,:)=0;
              h = real(ifft2(double(F)));
               else
            app.GrayscaleSwitch.Value='On';
            image(app,im)
                end
            else
            end

            % Notch filter column
            if Nfc1>0
               if strcmp(gray,'On') 
              F=fft2(double(h));
              F(:,Nfc1:1:Nfc2)=0;
              h = real(ifft2(double(F)));
               else
            app.GrayscaleSwitch.Value='On';
            image(app,im)
                end
            else
            end
            
            %Gaussian noise
            if strcmp(gn,'On')
                hf=h;
                h=imnoise(h,'gaussian');
                hn=h;
            else
            end
            
            %poisson noise
            if strcmp(pn,'On')
                hf=h;
                h=imnoise(h,'poisson');
                hn=h;
            else
            end
            
            %Salt&Pepper noise
            if strcmp(spn,'On')
                hf=h;
                h=imnoise(h,'salt & pepper');
                hn=h;
            else
            end
            
            %Speckle noise
            if strcmp(sn,'On')
                hf=h;
                h=imnoise(h,'speckle');
                hn=h;
            else
            end
            
            %Arithmatic mean
            if armm>0.1
                h=amean(h,[armm armn]);
            end
            
            %Harmonic mean
            if hmm>0.1
                h=hmean(h,hmm,hmn);
            end
            
           
            
            %Median filter
            if mfm>0.1
                [~,~,c]=size(h);
                if c>1
                red=h(:,:,1);
                red=medfilt2(red,[mfm mfn]);
                green=h(:,:,2);
                green=medfilt2(green,[mfm mfn]);
                blue=h(:,:,3);
                blue=medfilt2(blue,[mfm mfn]);
                h=cat(3,red,green,blue);
                else
                h=medfilt2(h,[mfm mfn]);
                end
            end
            
            %Min filter
            if mnf>2.5
               [~,~,c]=size(h);
               minf=@(x)min(x(:));
               if c>1
               red=h(:,:,1);
               red=nlfilter(red,[mnf mnf],minf);
               green=h(:,:,2);
               green=nlfilter(green,[mnf mnf],minf);
               blue=h(:,:,3);
               blue=nlfilter(blue,[mnf mnf],minf);
               h=cat(3,red,green,blue);
               else
               h=nlfilter(h,[mnf mnf],minf);    
               end
            end
            
            %Max filter
            if mxf>2.5
               [~,~,c]=size(h);
               maxf=@(x)max(x(:));
               if c>1
               red=h(:,:,1);
               red=nlfilter(red,[mxf mxf],maxf);
               green=h(:,:,2);
               green=nlfilter(green,[mxf mxf],maxf);
               blue=h(:,:,3);
               blue=nlfilter(blue,[mxf mxf],maxf);
               h=cat(3,red,green,blue);
               else
               h=nlfilter(h,[mxf mxf],maxf);    
               end
            end
            
            %Radon transform
            
            if rtn>0
                [~,~,c]=size(h);
                if c>1 
                app.GrayscaleSwitch.Value='On';
                image(app,im);
                else
                Radh = radon(h,0:rtm:rtn);
                Radh = Radh-min(min(Radh));
                Radh = Radh/max(max(abs(Radh)));
                
                rdfi={app.InterpolationDropDown.Value};
                rdff={app.FilterDropDown.Value};
                
                Bradh = iradon(Radh,0:irtm:irtn,app.InterpolationDropDown.Value,app.FilterDropDown.Value);
                Bradh = Bradh-min(min(Bradh));
                h = Bradh/max(max(abs(Bradh)));
            
                imagesc(app.RT,Radh)
                end
            end
            
            %Wiener filter
            if strcmp(wf,'On')
                noise=hn-hf;
                Sn=abs(fft2(noise)).^2;
                Sf= abs(fft2(im)).^2;
                NCORR=fftshift(ifft2(Sn));
                ICORR=fftshift(ifft2(Sf));
                h=deconvwnr(hn,PSF,NCORR,ICORR);
            else
            end
            

            
            %Iterative Non-Linear Restoration
             if strcmp(app.IterativeNonLinearRestorationIterationSwitch.Value,'On')
                h=deconvlucy(hn,PSF,iteration);
            else
             end
             
             
            % DFT Axes
              F=fft2(h);
              Fc=fftshift(F);  
            
            imagesc(app.DFTAxes,abs(Fc));
            imagesc(app.RI,h)
            plot(app.ImageHist,imhist(h));
            
            
            
            
        end
        
        function affine(app,~)
        global h tform
        h = imwarp(h,tform);
        imagesc(app.RI,h)
        plot(app.ImageHist,imhist(h));
        end

    

        end
    

    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % Configure image axes
            global im
            app.ImageAxes.Visible = 'off';
            app.ImageAxes.Colormap = gray(256);
            axis(app.ImageAxes, 'image');
            app.RI.Visible = 'off';
            app.RI.Colormap = gray(256);
            axis(app.RI, 'image');
            app.DFTAxes.Visible = 'off';
            app.DFTAxes.Colormap = gray(256);
            axis(app.DFTAxes, 'image');
            app.RT.Visible = 'off';
            app.RT.Colormap = gray(256);
            axis(app.RT, 'image');
            
            
            % Initial state
            app.cl.Value=0.01;
            app.GammaSlider.Value=1;
            app.SpecklenoiseSwitch.Value='Off';
            app.GaussiannoiseSwitch.Value='Off';
            app.PoissonnoiseSwitch.Value='Off';
            app.SaltPeppernoiseSwitch.Value='Off';
            app.EqualizationSwitch.Value='Off';
            app.GrayscaleSwitch.Value='Off';
            app.PrewittFliterSwitch.Value='Off';
            app.SobelFilterSwitch.Value='Off';
            app.WeinerfilterSwitch.Value='Off';
            plot(app.ImageHist,imhist(im));
            app.Ls.Value=0;
            app.Cst.Value=0;
            app.Afs1.Value=0;
            app.Afs2.Value=0;
            app.Dfr.Value=0;
            app.Gfs1.Value=0;
            app.Gfs2.Value=0;
            app.Gfg.Value=0;
            app.LoGfs1.Value=0;
            app.LoGfs2.Value=0;
            app.LoGfg.Value=0;
            app.tgv.Value=0;
            app.Mfl.Value=0;
            app.Mft.Value=0;
            app.Lfa.Value=0;
            app.Usfr.Value=0;
            app.Usfa.Value=0;
            app.Ihp.Value=0;
            app.Ilp.Value=0;
            app.Bwhf.Value=0;
            app.Bwho.Value=0;
            app.Bwlf.Value=0;
            app.Bwlo.Value=0;
            app.Ibpw.Value=0;
            app.Ibpr.Value=0;
            app.Ibrw.Value=0;
            app.Ibrr.Value=0;
            app.Nfr1.Value=0;
            app.Nfc1.Value=0;
            app.Nfr2.Value=0;
            app.Nfc2.Value=0;
            app.armm.Value=0;
            app.armn.Value=0;
            app.hmm.Value=0;
            app.hmn.Value=0;
            app.mfm.Value=0;
            app.mfn.Value=0;
            app.mnf.Value=0;
            app.mxf.Value=0;
            app.rtm.Value=0;
            app.rtn.Value=0;
            app.irtm.Value=0;
            app.irtn.Value=0;
            
            % Update the image and histograms
            updateimage(app, im); 
        end

        % Callback function
        function DropDownValueChanged(app, event)
            
            % Update the image and histograms
            updateimage(app, app.InterpolationDropDown.Value);
        end

        % Button pushed function: SelectimagefileButton
        function SelectimagefileButtonPushed(app, event)
               
            % Display uigetfile dialog
            filterspec = {'*.jpg;*.tif;*.png;*.gif','All Image Files'};
            [f, p] = uigetfile(filterspec);
            
            % Make sure user didn't cancel uigetfile dialog
            if (ischar(p))
               fname = [p f];
               startupFcn(app)
               updateimage(app, fname);
               h=imread(fname);
               F=fft2(h);
               Fc=fftshift(F);

            imagesc(app.DFTAxes,abs(Fc));
              
            plot(app.ImageHist,imhist(h))
            
            end
            
        end

        % Button pushed function: ImageinformationButton
        function ImageinformationButtonPushed(app, event)
            imageinfo(app.ImageAxes);     
        end

        % Value changed function: GammaSlider
        function GammaSliderValueChanged(app, event)
        global im
        image(app,im)
        end

        % Value changed function: GrayscaleSwitch
        function GrayscaleSwitchValueChanged(app, event)
            global im
        image(app,im)
            
        end

        % Value changed function: EqualizationSwitch
        function EqualizationSwitchValueChanged(app, event)
        global im
        image(app,im)
        end

        % Button pushed function: AllResetButton
        function AllResetButtonPushed(app, event)
            global im
            app.RT.Visible = 'off';
            app.DFTAxes.Visible = 'off';
            app.ImageHist.Visible = 'off';
            startupFcn(app)
            imagesc(app.RI,im);
        end

        % Value changed function: Afs1
        function Afs1ValueChanged(app, event)
            global Afs1
            Afs1 = app.Afs1.Value;            
        end

        % Value changed function: Afs2
        function Afs2ValueChanged(app, event)
            global Afs2
            Afs2 = app.Afs2.Value; 
            
        end

        % Button pushed function: GoButton
        function GoButtonPushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton
        function ResetButtonPushed(app, event)
          global Afs1 im
          Afs1=0;
          app.Afs1.Value=0;
          app.Afs2.Value=0;
          image(app,im)
        end

        % Value changed function: Ls
        function LsValueChanged(app, event)
            global ls
            ls = app.Ls.Value;            
        end

        % Button pushed function: GoButton_3
        function GoButton_3Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_3
        function ResetButton_3Pushed(app, event)
            global ls im
          ls=0;
          app.Ls.Value=0;
          image(app,im)
        end

        % Value changed function: Gfs1
        function Gfs1ValueChanged(app, event)
            global Gfs1
            Gfs1 = app.Gfs1.Value;
        end

        % Value changed function: Gfs2
        function Gfs2ValueChanged(app, event)
            global Gfs2
            Gfs2 = app.Gfs2.Value;
        end

        % Button pushed function: ResetButton_2
        function ResetButton_2Pushed(app, event)
            global Gfs1 im
          Gfs1=0;
          app.Gfs1.Value=0;
          app.Gfs2.Value=0;
          app.Gfg.Value=0;
          image(app,im)
        end

        % Value changed function: Gfg
        function GfgValueChanged(app, event)
            global Gfg
            Gfg = app.Gfg.Value;
        end

        % Button pushed function: GoButton_2
        function GoButton_2Pushed(app, event)
            global im
            image(app,im)
        end

        % Value changed function: Cst
        function CstValueChanged(app, event)
            global Cst
            Cst = app.Cst.Value;
            
        end

        % Button pushed function: GoButton_4
        function GoButton_4Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_4
        function ResetButton_4Pushed(app, event)
            global Cst im
          Cst=0;
          app.Cst.Value=0;
          image(app,im)
        end

        % Value changed function: Dfr
        function DfrValueChanged(app, event)
            global Dfr
            Dfr = app.Dfr.Value;
            
        end

        % Button pushed function: GoButton_5
        function GoButton_5Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_5
        function ResetButton_5Pushed(app, event)
            global Dfr im
          Dfr=0;
          app.Dfr.Value=0;
          image(app,im)
        end

        % Value changed function: LoGfs1
        function LoGfs1ValueChanged(app, event)
            global LoGfs1
            LoGfs1 = app.LoGfs1.Value;
            
        end

        % Value changed function: LoGfs2
        function LoGfs2ValueChanged(app, event)
            global LoGfs2
            LoGfs2 = app.LoGfs2.Value;
            
        end

        % Value changed function: LoGfg
        function LoGfgValueChanged(app, event)
            global LoGfg
            LoGfg = app.LoGfg.Value;
            
        end

        % Button pushed function: GoButton_7
        function GoButton_7Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_7
        function ResetButton_7Pushed(app, event)
            global LoGfg im
          LoGfg=0;  
          app.LoGfs1.Value=0;
          app.LoGfs2.Value=0;
          app.LoGfg.Value=0;
          image(app,im)
        end

        % Button pushed function: gamma1Button
        function gamma1ButtonPushed(app, event)
           global im
            app.GammaSlider.Value=1;
            image(app,im)
        end

        % Value changed function: tgv
        function tgvValueChanged(app, event)
           global im
            app.GammaSlider.Value=app.tgv.Value;
            image(app,im)
        end

        % Button pushed function: gamma1Button_4
        function gamma1Button_4Pushed(app, event)
            global im
            app.GammaSlider.Value=2;
            image(app,im)
        end

        % Button pushed function: gamma1Button_2
        function gamma1Button_2Pushed(app, event)
            global im
            app.GammaSlider.Value=1.5;
            image(app,im)
        end

        % Button pushed function: gamma1Button_3
        function gamma1Button_3Pushed(app, event)
            global im
            app.GammaSlider.Value=0.5;
            image(app,im)
        end

        % Button pushed function: gamma1Button_5
        function gamma1Button_5Pushed(app, event)
            global im
            app.GammaSlider.Value=0;
            image(app,im)
        end

        % Value changed function: Mfl
        function MflValueChanged(app, event)
            global Mfl
            Mfl = app.Mfl.Value;
            
        end

        % Value changed function: Mft
        function MftValueChanged(app, event)
            global Mft
            Mft = app.Mft.Value;
            
        end

        % Button pushed function: GoButton_6
        function GoButton_6Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_6
        function ResetButton_6Pushed(app, event)
            global Mft im
          Mft=0;  
          app.Mfl.Value=0;
          app.Mft.Value=0;
          image(app,im)
        end

        % Value changed function: PrewittFliterSwitch
        function PrewittFliterSwitchValueChanged(app, event)
            global im
        image(app,im)
        end

        % Value changed function: SobelFilterSwitch
        function SobelFilterSwitchValueChanged(app, event)
            global im
        image(app,im)
            
        end

        % Callback function
        function PaddingOptionButtonGroupSelectionChanged(app, event)

        end

        % Value changed function: Lfa
        function LfaValueChanged(app, event)
            global Lfa
            Lfa = app.Lfa.Value;
            
        end

        % Button pushed function: GoButton_8
        function GoButton_8Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_8
        function ResetButton_8Pushed(app, event)
            global Lfa im
          Lfa=0;  
          app.Lfa.Value=0;
          image(app,im)
        end

        % Value changed function: Usfr
        function UsfrValueChanged(app, event)
            global Usfr
            Usfr = app.Usfr.Value;
            
        end

        % Value changed function: Usfa
        function UsfaValueChanged(app, event)
            global Usfa
            Usfa = app.Usfa.Value;
            
        end

        % Button pushed function: GoButton_9
        function GoButton_9Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_9
        function ResetButton_9Pushed(app, event)
            global Usfr im
            app.Usfr.Value=0;
            app.Usfa.Value=0;
            Usfr=0;
            image(app,im)
        end

        % Value changed function: Ihp
        function IhpValueChanged(app, event)
            global Ihp
            Ihp = app.Ihp.Value;
            
        end

        % Button pushed function: GoButton_10
        function GoButton_10Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_10
        function ResetButton_10Pushed(app, event)
            global Ihp im
            app.Ihp.Value=0;
            Ihp=0;
            image(app,im)
        end

        % Value changed function: Ilp
        function IlpValueChanged(app, event)
            global Ilp
            Ilp = app.Ilp.Value;
            
        end

        % Button pushed function: GoButton_11
        function GoButton_11Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_11
        function ResetButton_11Pushed(app, event)
            global Ilp im
            app.Ilp.Value=0;
            Ilp=0;
            image(app,im)
        end

        % Value changed function: Bwhf
        function BwhfValueChanged(app, event)
            global Bwhf
            Bwhf = app.Bwhf.Value;  
        end

        % Button pushed function: GoButton_12
        function GoButton_12Pushed(app, event)
            global im
            image(app,im)
        end

        % Value changed function: Bwho
        function BwhoValueChanged(app, event)
            global Bwfo
            Bwfo = app.Bwho.Value;
            
        end

        % Button pushed function: ResetButton_12
        function ResetButton_12Pushed(app, event)
            global Bwhf im Bwfo
            app.Bwhf.Value=0;
            app.Bwho.Value=0;
            Bwhf=0;
            Bwfo=0;
            image(app,im)
        end

        % Value changed function: Bwlf
        function BwlfValueChanged(app, event)
            global Bwlf
            Bwlf = app.Bwlf.Value;
            
        end

        % Value changed function: Bwlo
        function BwloValueChanged(app, event)
            global Bwlo
            Bwlo = app.Bwlo.Value;
            
        end

        % Button pushed function: GoButton_13
        function GoButton_13Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_13
        function ResetButton_13Pushed(app, event)
            global Bwlo  Bwlf im 
            app.Bwlf.Value=0;
            app.Bwlo.Value=0;
            Bwlo=0;
            Bwlf=0;
            image(app,im)
        end

        % Value changed function: Ibpr
        function IbprValueChanged(app, event)
           global Ibpr
            Ibpr = app.Ibpr.Value;
            
        end

        % Value changed function: Ibpw
        function IbpwValueChanged(app, event)
            global Ibpw
            Ibpw = app.Ibpw.Value;
            
        end

        % Button pushed function: GoButton_14
        function GoButton_14Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_14
        function ResetButton_14Pushed(app, event)
            global Ibpr Ibpw im 
            app.Ibpw.Value=0;
            app.Ibpr.Value=0;
            Ibpr=0;
            Ibpw=0;
            image(app,im)
        end

        % Value changed function: Ibrr
        function IbrrValueChanged(app, event)
            global Ibrr
            Ibrr = app.Ibrr.Value;
            
        end

        % Value changed function: Ibrw
        function IbrwValueChanged(app, event)
            global Ibrw
            Ibrw = app.Ibrw.Value;
            
        end

        % Value changed function: Nfr1
        function Nfr1ValueChanged(app, event)
            global Nfr1
            Nfr1 = app.Nfr1.Value;
            
        end

        % Button pushed function: ResetButton_16
        function ResetButton_16Pushed(app, event)
            global Nfr1 Nfr2 im
            app.Nfr1.Value=0;
            app.Nfr2.Value=0;
            Nfr1=0;
            Nfr2 = 0;
            image(app,im)
        end

        % Button pushed function: GoButton_16
        function GoButton_16Pushed(app, event)
            global im
            image(app,im)
        end

        % Value changed function: Nfc1
        function Nfc1ValueChanged(app, event)
            global Nfc1
            Nfc1 = app.Nfc1.Value;
            
        end

        % Button pushed function: GoButton_17
        function GoButton_17Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_17
        function ResetButton_17Pushed(app, event)
            global Nfc1 Nfc2 im
            app.Nfc1.Value=0;
            app.Nfc2.Value=0;
            Nfc1=0;
            Nfc2=0;
            image(app,im)
        end

        % Button pushed function: GoButton_15
        function GoButton_15Pushed(app, event)
            global im
            image(app,im)
        end

        % Button pushed function: ResetButton_15
        function ResetButton_15Pushed(app, event)
            global Ibrr Ibrw im 
            app.Ibrw.Value=0;
            app.Ibrr.Value=0;
            Ibrr=0;
            Ibrw=0;
            image(app,im)
        end

        % Button pushed function: SaveasButton
        function SaveasButtonPushed(app, event)
            global h
            startingFolder = userpath;
            defaultFileName = fullfile(startingFolder, '*.*');
            [baseFileName, folder] = uiputfile(defaultFileName, 'Specify a file');
            if baseFileName == 0
            return;
            end
            fullFileName = fullfile(folder, baseFileName);
            imwrite(h, fullFileName);
        end

        % Value changed function: Nfr2
        function Nfr2ValueChanged(app, event)
          global Nfr2
            Nfr2 = app.Nfr2.Value;
            
        end

        % Value changed function: Nfc2
        function Nfc2ValueChanged(app, event)
            global Nfc2
            Nfc2 = app.Nfc1.Value;
            
        end

        % Callback function
        function DropDownValueChanged2(app, event)
            global im
            image(app,im)
            
        end

        % Value changed function: GaussiannoiseSwitch
        function GaussiannoiseSwitchValueChanged(app, event)
            global im
        image(app,im)
            
        end

        % Value changed function: PoissonnoiseSwitch
        function PoissonnoiseSwitchValueChanged(app, event)
global im
        image(app,im)
        end

        % Value changed function: SaltPeppernoiseSwitch
        function SaltPeppernoiseSwitchValueChanged(app, event)
global im
        image(app,im)
        end

        % Value changed function: SpecklenoiseSwitch
        function SpecklenoiseSwitchValueChanged(app, event)
global im
        image(app,im)
        end

        % Value changed function: armm
        function armmValueChanged(app, event)
            global armm
            armm=app.armm.Value;         
        end

        % Value changed function: armn
        function armnValueChanged(app, event)
            global armn
            armn=app.armn.Value;   
        end

        % Button pushed function: GoButton_18
        function GoButton_18Pushed(app, event)
global im
        image(app,im)
        end

        % Button pushed function: ResetButton_18
        function ResetButton_18Pushed(app, event)
            global armn armm im
            app.armm.Value=0;
            app.armn.Value=0;
            armn=0;
            armm=0;
            image(app,im)
        end

        % Value changed function: hmm
        function hmmValueChanged(app, event)
            global hmm
            hmm=app.hmm.Value; 
        end

        % Value changed function: hmn
        function hmnValueChanged(app, event)
            global hmn
            hmn=app.hmn.Value; 
        end

        % Callback function
        function chmmValueChanged(app, event)
            global chmm
            chmm=app.armm.Value; 
        end

        % Callback function
        function chmnValueChanged(app, event)
            global chmn
            chmn=app.armm.Value; 
        end

        % Button pushed function: GoButton_19
        function GoButton_19Pushed(app, event)
        global im
        image(app,im)    
        end

        % Callback function
        function GoButton_20Pushed(app, event)
          global im
        image(app,im)  
        end

        % Button pushed function: ResetButton_19
        function ResetButton_19Pushed(app, event)
            global hmm hmn im
            app.hmm.Value=0;
            app.hmn.Value=0;
            hmn=0;
            hmm=0;
            image(app,im)
        end

        % Callback function
        function ResetButton_20Pushed(app, event)
            global chmn chmm chmq im
            app.chmm.Value=0;
            app.chmn.Value=0;
            app.chmq.Value=0;
            chmn=0;
            chmm=0;
            chmq=0;
            image(app,im)
        end

        % Callback function
        function chmqValueChanged(app, event)
            global chmq
            chmq=app.chmq.Value; 
        end

        % Value changed function: mfm
        function mfmValueChanged(app, event)
            global mfm
            mfm = app.mfm.Value;
            
        end

        % Value changed function: mfn
        function mfnValueChanged(app, event)
            global mfn
            mfn = app.mfn.Value;
            
        end

        % Button pushed function: GoButton_21
        function GoButton_21Pushed(app, event)
               global im
        image(app,im)      
        end

        % Button pushed function: ResetButton_21
        function ResetButton_21Pushed(app, event)
            global mfm mfn im
            app.mfm.Value=0;
            app.mfn.Value=0;
            mfn=0;
            mfm=0;
            image(app,im)
        end

        % Value changed function: mxf
        function mxfValueChanged(app, event)
                global mxf
                mxf=app.mxf.Value;
        end

        % Value changed function: mnf
        function mnfValueChanged(app, event)
            global mnf
                mnf=app.mnf.Value;
        end

        % Button pushed function: GoButton_22
        function GoButton_22Pushed(app, event)
            global im
        image(app,im)  
        end

        % Button pushed function: GoButton_23
        function GoButton_23Pushed(app, event)
            global im
        image(app,im)  
        end

        % Button pushed function: ResetButton_22
        function ResetButton_22Pushed(app, event)
            global mxf im
            app.mxf.Value=0;
            mxf=0;
            image(app,im)
        end

        % Button pushed function: ResetButton_23
        function ResetButton_23Pushed(app, event)
            global mnf im
            app.mnf.Value=0;
            mnf=0;
            image(app,im)
        end

        % Value changed function: rtm
        function rtmValueChanged(app, event)
            global rtm
            rtm = app.rtm.Value;
            
        end

        % Value changed function: rtn
        function rtnValueChanged(app, event)
            global rtn
            rtn = app.rtn.Value;
            
        end

        % Value changed function: irtm
        function irtmValueChanged(app, event)
            global irtm
            irtm = app.irtm.Value;
            
        end

        % Value changed function: irtn
        function irtnValueChanged(app, event)
            global irtn
            irtn = app.irtn.Value;
            
        end

        % Button pushed function: GoButton_24
        function GoButton_24Pushed(app, event)
            global im
            image(app,im);
        end

        % Button pushed function: ResetButton_24
        function ResetButton_24Pushed(app, event)
            global im rtn rtm irtm irtn 
            app.rtm.Value=0;
            app.rtn.Value=0;
            app.irtm.Value=0;
            app.irtn.Value=0;
            rtm=0;
            rtn=0;
            irtm=0;
            irtn=0;
            app.RT.Visible = 'off';
            image(app,im);
        end

        % Value changed function: InterpolationDropDown
        function InterpolationDropDownValueChanged(app, event)
            global rdfi
            rdfi = app.InterpolationDropDown.Value;
            
        end

        % Value changed function: FilterDropDown
        function FilterDropDownValueChanged(app, event)
            global rdff
            rdff = app.FilterDropDown.Value;
            
        end

        % Button pushed function: GoButton_25
        function GoButton_25Pushed(app, event)
            global scalingx scalingy tform h
            tform= affine2d([scalingx 0 0; 0 scalingy 0;0 0 1]);
            affine(app,h)
            app.scalingx.Value=0;
            app.scalingy.Value=0;
            scalingx=0;
            scalingy=0;
            tform=affine2d([1 0 0; 0 1 0;0 0 1]);
        end

        % Value changed function: scalingx
        function scalingxValueChanged(app, event)
            global scalingx
            scalingx = app.scalingx.Value;
            
        end

        % Value changed function: scalingy
        function scalingyValueChanged(app, event)
            global scalingy
            scalingy = app.scalingy.Value;
            
        end

        % Value changed function: rotation
        function rotationValueChanged(app, event)
            global theta 
            theta=app.rotation.Value;
           
            
        end

        % Button pushed function: GoButton_26
        function GoButton_26Pushed(app, event)
            global theta tform h
            tform=affine2d([cosd(theta) sind(theta) 0;-sind(theta) cosd(theta) 0;0 0 1]);
            affine(app,h)
            app.rotation.Value=0;
            theta=0;
            tform=affine2d([1 0 0; 0 1 0;0 0 1]);
        end

        % Value changed function: shearh
        function shearhValueChanged(app, event)
            global shearh
            shearh = app.shearh.Value;
            
        end

        % Value changed function: shearv
        function shearvValueChanged(app, event)
            global shearv
            shearv = app.shearv.Value;
            
        end

        % Button pushed function: GoButton_27
        function GoButton_27Pushed(app, event)
            global shearh h tform
            tform=affine2d([1 0 0;shearh 1 0;0 0 1]);
            affine(app,h)
            app.shearh.Value=0;
            shearh=0;
            tform=affine2d([1 0 0; 0 1 0;0 0 1]);
        end

        % Button pushed function: GoButton_28
        function GoButton_28Pushed(app, event)
            global shearv h tform
            tform=affine2d([1 shearv 0;0 1 0;0 0 1]);
            affine(app,h)
            app.shearv.Value=0;
            shearv=0;
            tform=affine2d([1 0 0; 0 1 0;0 0 1]);
        end

        % Value changed function: dx
        function dxValueChanged(app, event)
            global dx
            dx = app.dx.Value;
            
        end

        % Value changed function: dy
        function dyValueChanged(app, event)
            global dy
            dy = app.dy.Value;
            
        end

        % Button pushed function: GoButton_29
        function GoButton_29Pushed(app, event)
            global dx dy h tform
            tform=affine2d([1 0 0;0 1 0;dx dy 1]);
            affine(app,h)
            app.dx.Value=0;
            app.dy.Value=0;
            dx=0;
            dy=0;
            tform=affine2d([1 0 0; 0 1 0;0 0 1]);
        end

        % Button pushed function: GoButton_30
        function GoButton_30Pushed(app, event)
            global tform h
            tform=affine2d([1 0 0; 0 -1 0; 0 0 1]);
            affine(app,h)
            tform=affine2d([1 0 0; 0 1 0;0 0 1]);
        end

        % Button pushed function: ResetButton_25
        function ResetButton_25Pushed(app, event)
            global im
            image(app,im);
        end

        % Value changed function: WeinerfilterSwitch
        function WeinerfilterSwitchValueChanged(app, event)
            global im
        image(app,im)
        end

        % Value changed function: cl
        function clValueChanged(app, event)
            global cl im
            cl = app.cl.Value;
            image(app,im)
        end

        % Value changed function: RangeDropDown
        function RangeDropDownValueChanged(app, event)
global im
        image(app,im)
            
        end

        % Value changed function: DistributionDropDown
        function DistributionDropDownValueChanged(app, event)
global im
        image(app,im)
        end

        % Value changed function: ahe
        function aheValueChanged(app, event)
            global im
        image(app,im)
            
        end

        % Value changed function: iteration
        function iterationValueChanged(app, event)
            global iteration
            iteration = app.iteration.Value;
            
        end

        % Value changed function: 
        % IterativeNonLinearRestorationIterationSwitch
        function IterativeNonLinearRestorationIterationSwitchValueChanged(app, event)
            global im
        image(app,im)
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Color = [0.9686 0.9686 0.9686];
            app.UIFigure.Position = [100 100 1614 775];
            app.UIFigure.Name = 'Image Histograms';
            app.UIFigure.Resize = 'off';

            % Create ImageAxes
            app.ImageAxes = uiaxes(app.UIFigure);
            app.ImageAxes.XTick = [];
            app.ImageAxes.XTickLabel = {'[ ]'};
            app.ImageAxes.YTick = [];
            app.ImageAxes.Position = [43 381 479 352];

            % Create RI
            app.RI = uiaxes(app.UIFigure);
            app.RI.XTick = [];
            app.RI.XTickLabel = {'[ ]'};
            app.RI.YTick = [];
            app.RI.Position = [834 339 698 398];

            % Create BlueAxes
            app.BlueAxes = uiaxes(app.UIFigure);
            title(app.BlueAxes, 'Blue')
            xlabel(app.BlueAxes, 'Intensity')
            ylabel(app.BlueAxes, 'Pixels')
            app.BlueAxes.XLim = [0 255];
            app.BlueAxes.XTick = [0 128 255];
            app.BlueAxes.Position = [521 627 175 92];

            % Create GreenAxes
            app.GreenAxes = uiaxes(app.UIFigure);
            title(app.GreenAxes, 'Green')
            xlabel(app.GreenAxes, 'Intensity')
            ylabel(app.GreenAxes, 'Pixels')
            app.GreenAxes.XLim = [0 255];
            app.GreenAxes.XTick = [0 128 255];
            app.GreenAxes.Position = [521 381 176 92];

            % Create RedAxes
            app.RedAxes = uiaxes(app.UIFigure);
            title(app.RedAxes, 'Red')
            xlabel(app.RedAxes, 'Intensity')
            ylabel(app.RedAxes, 'Pixels')
            app.RedAxes.XLim = [0 255];
            app.RedAxes.XTick = [0 128 255];
            app.RedAxes.Position = [520 499 175 93];

            % Create ImageHist
            app.ImageHist = uiaxes(app.UIFigure);
            app.ImageHist.XTick = [];
            app.ImageHist.XTickLabel = {'[ ]'};
            app.ImageHist.YTick = [];
            app.ImageHist.Position = [737 18 435 273];

            % Create DFTAxes
            app.DFTAxes = uiaxes(app.UIFigure);
            app.DFTAxes.XTick = [];
            app.DFTAxes.XTickLabel = {'[ ]'};
            app.DFTAxes.YTick = [];
            app.DFTAxes.Position = [1171 18 410 274];

            % Create SelectimagefileButton
            app.SelectimagefileButton = uibutton(app.UIFigure, 'push');
            app.SelectimagefileButton.ButtonPushedFcn = createCallbackFcn(app, @SelectimagefileButtonPushed, true);
            app.SelectimagefileButton.FontWeight = 'bold';
            app.SelectimagefileButton.Position = [159 354 116 22];
            app.SelectimagefileButton.Text = 'Select image file';

            % Create ImageinformationButton
            app.ImageinformationButton = uibutton(app.UIFigure, 'push');
            app.ImageinformationButton.ButtonPushedFcn = createCallbackFcn(app, @ImageinformationButtonPushed, true);
            app.ImageinformationButton.FontWeight = 'bold';
            app.ImageinformationButton.Position = [296 354 120 22];
            app.ImageinformationButton.Text = 'Image information';

            % Create OriginalImageLabel
            app.OriginalImageLabel = uilabel(app.UIFigure);
            app.OriginalImageLabel.FontSize = 20;
            app.OriginalImageLabel.FontWeight = 'bold';
            app.OriginalImageLabel.Position = [219 731 144 24];
            app.OriginalImageLabel.Text = 'Original Image';

            % Create GammaSlider
            app.GammaSlider = uislider(app.UIFigure);
            app.GammaSlider.Limits = [0 2];
            app.GammaSlider.MajorTicks = [0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1 1.05 1.1 1.15 1.2 1.25 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2];
            app.GammaSlider.MajorTickLabels = {'0', '', '', '', '', '', '', '', '', '', '0.5', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '1.5', '', '', '', '', '', '', '', '', '', '2.'};
            app.GammaSlider.Orientation = 'vertical';
            app.GammaSlider.ValueChangedFcn = createCallbackFcn(app, @GammaSliderValueChanged, true);
            app.GammaSlider.MinorTicks = [];
            app.GammaSlider.Position = [771 363 7 364];
            app.GammaSlider.Value = 0.1;

            % Create EqualizationSwitchLabel
            app.EqualizationSwitchLabel = uilabel(app.UIFigure);
            app.EqualizationSwitchLabel.HorizontalAlignment = 'center';
            app.EqualizationSwitchLabel.FontWeight = 'bold';
            app.EqualizationSwitchLabel.Position = [747 291 76 22];
            app.EqualizationSwitchLabel.Text = 'Equalization';

            % Create EqualizationSwitch
            app.EqualizationSwitch = uiswitch(app.UIFigure, 'slider');
            app.EqualizationSwitch.ValueChangedFcn = createCallbackFcn(app, @EqualizationSwitchValueChanged, true);
            app.EqualizationSwitch.Position = [759 311 45 20];

            % Create ResultImageLabel
            app.ResultImageLabel = uilabel(app.UIFigure);
            app.ResultImageLabel.FontSize = 20;
            app.ResultImageLabel.FontWeight = 'bold';
            app.ResultImageLabel.Position = [1118 732 130 24];
            app.ResultImageLabel.Text = 'Result Image';

            % Create IntensityGammaLabel
            app.IntensityGammaLabel = uilabel(app.UIFigure);
            app.IntensityGammaLabel.FontWeight = 'bold';
            app.IntensityGammaLabel.Position = [734 732 102 22];
            app.IntensityGammaLabel.Text = 'Intensity Gamma';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.AutoResizeChildren = 'off';
            app.TabGroup.TabLocation = 'left';
            app.TabGroup.Position = [44 23 684 309];

            % Create DRMTab
            app.DRMTab = uitab(app.TabGroup);
            app.DRMTab.AutoResizeChildren = 'off';
            app.DRMTab.Title = 'DRM';

            % Create DynamicRangeManipulationLabel
            app.DynamicRangeManipulationLabel = uilabel(app.DRMTab);
            app.DynamicRangeManipulationLabel.FontWeight = 'bold';
            app.DynamicRangeManipulationLabel.Position = [213 277 173 22];
            app.DynamicRangeManipulationLabel.Text = 'Dynamic Range Manipulation';

            % Create LogarithmicTransformationLabel
            app.LogarithmicTransformationLabel = uilabel(app.DRMTab);
            app.LogarithmicTransformationLabel.Position = [20 232 151 22];
            app.LogarithmicTransformationLabel.Text = 'Logarithmic Transformation';

            % Create gcxlogfLabel
            app.gcxlogfLabel = uilabel(app.DRMTab);
            app.gcxlogfLabel.FontSize = 15;
            app.gcxlogfLabel.Position = [64 202 135 25];
            app.gcxlogfLabel.Text = 'g=c x log(          +f)';

            % Create Ls
            app.Ls = uieditfield(app.DRMTab, 'numeric');
            app.Ls.ValueChangedFcn = createCallbackFcn(app, @LsValueChanged, true);
            app.Ls.Position = [132 200 34 23];

            % Create GoButton_3
            app.GoButton_3 = uibutton(app.DRMTab, 'push');
            app.GoButton_3.ButtonPushedFcn = createCallbackFcn(app, @GoButton_3Pushed, true);
            app.GoButton_3.Position = [197 201 68 22];
            app.GoButton_3.Text = 'Go';

            % Create ResetButton_3
            app.ResetButton_3 = uibutton(app.DRMTab, 'push');
            app.ResetButton_3.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_3Pushed, true);
            app.ResetButton_3.Position = [276 201 55 22];
            app.ResetButton_3.Text = 'Reset';

            % Create ContrastStretchingTransformationLabel
            app.ContrastStretchingTransformationLabel = uilabel(app.DRMTab);
            app.ContrastStretchingTransformationLabel.Position = [19 143 193 22];
            app.ContrastStretchingTransformationLabel.Text = 'Contrast-Stretching Transformation';

            % Create Cst
            app.Cst = uieditfield(app.DRMTab, 'numeric');
            app.Cst.ValueChangedFcn = createCallbackFcn(app, @CstValueChanged, true);
            app.Cst.Position = [183 114 34 23];

            % Create GoButton_4
            app.GoButton_4 = uibutton(app.DRMTab, 'push');
            app.GoButton_4.ButtonPushedFcn = createCallbackFcn(app, @GoButton_4Pushed, true);
            app.GoButton_4.Position = [223 115 68 22];
            app.GoButton_4.Text = 'Go';

            % Create ResetButton_4
            app.ResetButton_4 = uibutton(app.DRMTab, 'push');
            app.ResetButton_4.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_4Pushed, true);
            app.ResetButton_4.Position = [302 115 55 22];
            app.ResetButton_4.Text = 'Reset';

            % Create krEELabel
            app.krEELabel = uilabel(app.DRMTab);
            app.krEELabel.FontSize = 15;
            app.krEELabel.Position = [64 83 120 51];
            app.krEELabel.Text = {'1/(1+(k/r)^E),  E='; ''; ''; ''};

            % Create MaskingTab
            app.MaskingTab = uitab(app.TabGroup);
            app.MaskingTab.AutoResizeChildren = 'off';
            app.MaskingTab.Title = 'Masking';
            app.MaskingTab.BackgroundColor = [0.9412 0.9412 0.9412];

            % Create AverageFiltersizeLabel
            app.AverageFiltersizeLabel = uilabel(app.MaskingTab);
            app.AverageFiltersizeLabel.HorizontalAlignment = 'right';
            app.AverageFiltersizeLabel.Position = [13 232 105 22];
            app.AverageFiltersizeLabel.Text = 'Average Filter size';

            % Create Afs1
            app.Afs1 = uieditfield(app.MaskingTab, 'numeric');
            app.Afs1.ValueChangedFcn = createCallbackFcn(app, @Afs1ValueChanged, true);
            app.Afs1.Position = [124 232 34 22];

            % Create XLabel
            app.XLabel = uilabel(app.MaskingTab);
            app.XLabel.Position = [162 232 10 22];
            app.XLabel.Text = 'X';

            % Create Afs2
            app.Afs2 = uieditfield(app.MaskingTab, 'numeric');
            app.Afs2.ValueChangedFcn = createCallbackFcn(app, @Afs2ValueChanged, true);
            app.Afs2.Position = [174 232 33 22];

            % Create GoButton
            app.GoButton = uibutton(app.MaskingTab, 'push');
            app.GoButton.ButtonPushedFcn = createCallbackFcn(app, @GoButtonPushed, true);
            app.GoButton.Position = [220 232 68 22];
            app.GoButton.Text = 'Go';

            % Create ResetButton
            app.ResetButton = uibutton(app.MaskingTab, 'push');
            app.ResetButton.ButtonPushedFcn = createCallbackFcn(app, @ResetButtonPushed, true);
            app.ResetButton.Position = [299 232 55 22];
            app.ResetButton.Text = 'Reset';

            % Create GaussianFiltersizeLabel
            app.GaussianFiltersizeLabel = uilabel(app.MaskingTab);
            app.GaussianFiltersizeLabel.HorizontalAlignment = 'right';
            app.GaussianFiltersizeLabel.Position = [12 154 111 22];
            app.GaussianFiltersizeLabel.Text = 'Gaussian Filter size';

            % Create Gfs1
            app.Gfs1 = uieditfield(app.MaskingTab, 'numeric');
            app.Gfs1.ValueChangedFcn = createCallbackFcn(app, @Gfs1ValueChanged, true);
            app.Gfs1.Position = [129 154 34 22];

            % Create XLabel_2
            app.XLabel_2 = uilabel(app.MaskingTab);
            app.XLabel_2.Position = [168 154 10 22];
            app.XLabel_2.Text = 'X';

            % Create Gfs2
            app.Gfs2 = uieditfield(app.MaskingTab, 'numeric');
            app.Gfs2.ValueChangedFcn = createCallbackFcn(app, @Gfs2ValueChanged, true);
            app.Gfs2.Position = [180 154 33 22];

            % Create GoButton_2
            app.GoButton_2 = uibutton(app.MaskingTab, 'push');
            app.GoButton_2.ButtonPushedFcn = createCallbackFcn(app, @GoButton_2Pushed, true);
            app.GoButton_2.Position = [308 154 68 22];
            app.GoButton_2.Text = 'Go';

            % Create ResetButton_2
            app.ResetButton_2 = uibutton(app.MaskingTab, 'push');
            app.ResetButton_2.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_2Pushed, true);
            app.ResetButton_2.Position = [387 154 55 22];
            app.ResetButton_2.Text = 'Reset';

            % Create sigmaLabel
            app.sigmaLabel = uilabel(app.MaskingTab);
            app.sigmaLabel.Position = [219 154 48 22];
            app.sigmaLabel.Text = ', sigma ';

            % Create Gfg
            app.Gfg = uieditfield(app.MaskingTab, 'numeric');
            app.Gfg.ValueChangedFcn = createCallbackFcn(app, @GfgValueChanged, true);
            app.Gfg.Position = [264 154 33 22];

            % Create SpatialFilteringLabel
            app.SpatialFilteringLabel = uilabel(app.MaskingTab);
            app.SpatialFilteringLabel.FontWeight = 'bold';
            app.SpatialFilteringLabel.Position = [245 277 102 22];
            app.SpatialFilteringLabel.Text = 'Spatial Filtering  ';

            % Create GoButton_5
            app.GoButton_5 = uibutton(app.MaskingTab, 'push');
            app.GoButton_5.ButtonPushedFcn = createCallbackFcn(app, @GoButton_5Pushed, true);
            app.GoButton_5.Position = [190 191 68 22];
            app.GoButton_5.Text = 'Go';

            % Create ResetButton_5
            app.ResetButton_5 = uibutton(app.MaskingTab, 'push');
            app.ResetButton_5.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_5Pushed, true);
            app.ResetButton_5.Position = [269 191 55 22];
            app.ResetButton_5.Text = 'Reset';

            % Create DiskFilterRaidussizeLabel
            app.DiskFilterRaidussizeLabel = uilabel(app.MaskingTab);
            app.DiskFilterRaidussizeLabel.HorizontalAlignment = 'right';
            app.DiskFilterRaidussizeLabel.Position = [13 191 124 22];
            app.DiskFilterRaidussizeLabel.Text = 'Disk Filter Raidus size';

            % Create Dfr
            app.Dfr = uieditfield(app.MaskingTab, 'numeric');
            app.Dfr.ValueChangedFcn = createCallbackFcn(app, @DfrValueChanged, true);
            app.Dfr.Position = [143 191 34 22];

            % Create GoButton_6
            app.GoButton_6 = uibutton(app.MaskingTab, 'push');
            app.GoButton_6.ButtonPushedFcn = createCallbackFcn(app, @GoButton_6Pushed, true);
            app.GoButton_6.Position = [252 44 68 22];
            app.GoButton_6.Text = 'Go';

            % Create ResetButton_6
            app.ResetButton_6 = uibutton(app.MaskingTab, 'push');
            app.ResetButton_6.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_6Pushed, true);
            app.ResetButton_6.Position = [331 44 55 22];
            app.ResetButton_6.Text = 'Reset';

            % Create ThetaLabel
            app.ThetaLabel = uilabel(app.MaskingTab);
            app.ThetaLabel.Position = [166 44 46 22];
            app.ThetaLabel.Text = ', Theta ';

            % Create Mft
            app.Mft = uieditfield(app.MaskingTab, 'numeric');
            app.Mft.ValueChangedFcn = createCallbackFcn(app, @MftValueChanged, true);
            app.Mft.Position = [208 44 33 22];

            % Create MotionFilterLinearLabel
            app.MotionFilterLinearLabel = uilabel(app.MaskingTab);
            app.MotionFilterLinearLabel.HorizontalAlignment = 'right';
            app.MotionFilterLinearLabel.Position = [13 44 108 22];
            app.MotionFilterLinearLabel.Text = 'Motion Filter Linear';

            % Create Mfl
            app.Mfl = uieditfield(app.MaskingTab, 'numeric');
            app.Mfl.ValueChangedFcn = createCallbackFcn(app, @MflValueChanged, true);
            app.Mfl.Position = [127 44 34 22];

            % Create XLabel_3
            app.XLabel_3 = uilabel(app.MaskingTab);
            app.XLabel_3.Position = [141 80 10 22];
            app.XLabel_3.Text = 'X';

            % Create LoGfs2
            app.LoGfs2 = uieditfield(app.MaskingTab, 'numeric');
            app.LoGfs2.ValueChangedFcn = createCallbackFcn(app, @LoGfs2ValueChanged, true);
            app.LoGfs2.Position = [153 80 33 22];

            % Create GoButton_7
            app.GoButton_7 = uibutton(app.MaskingTab, 'push');
            app.GoButton_7.ButtonPushedFcn = createCallbackFcn(app, @GoButton_7Pushed, true);
            app.GoButton_7.Position = [276 80 68 22];
            app.GoButton_7.Text = 'Go';

            % Create ResetButton_7
            app.ResetButton_7 = uibutton(app.MaskingTab, 'push');
            app.ResetButton_7.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_7Pushed, true);
            app.ResetButton_7.Position = [355 80 55 22];
            app.ResetButton_7.Text = 'Reset';

            % Create sigmaLabel_2
            app.sigmaLabel_2 = uilabel(app.MaskingTab);
            app.sigmaLabel_2.Position = [192 80 41 22];
            app.sigmaLabel_2.Text = ',sigma';

            % Create LoGfg
            app.LoGfg = uieditfield(app.MaskingTab, 'numeric');
            app.LoGfg.ValueChangedFcn = createCallbackFcn(app, @LoGfgValueChanged, true);
            app.LoGfg.Position = [232 80 33 22];

            % Create LoGFiltersizeLabel
            app.LoGFiltersizeLabel = uilabel(app.MaskingTab);
            app.LoGFiltersizeLabel.HorizontalAlignment = 'right';
            app.LoGFiltersizeLabel.Position = [13 80 83 22];
            app.LoGFiltersizeLabel.Text = 'LoG Filter size';

            % Create LoGfs1
            app.LoGfs1 = uieditfield(app.MaskingTab, 'numeric');
            app.LoGfs1.ValueChangedFcn = createCallbackFcn(app, @LoGfs1ValueChanged, true);
            app.LoGfs1.Position = [102 80 34 22];

            % Create PrewittFliterSwitchLabel
            app.PrewittFliterSwitchLabel = uilabel(app.MaskingTab);
            app.PrewittFliterSwitchLabel.HorizontalAlignment = 'center';
            app.PrewittFliterSwitchLabel.Position = [435 44 72 22];
            app.PrewittFliterSwitchLabel.Text = 'Prewitt Fliter';

            % Create PrewittFliterSwitch
            app.PrewittFliterSwitch = uiswitch(app.MaskingTab, 'slider');
            app.PrewittFliterSwitch.ValueChangedFcn = createCallbackFcn(app, @PrewittFliterSwitchValueChanged, true);
            app.PrewittFliterSwitch.Position = [531 44 45 20];

            % Create SobelFilterSwitchLabel
            app.SobelFilterSwitchLabel = uilabel(app.MaskingTab);
            app.SobelFilterSwitchLabel.HorizontalAlignment = 'center';
            app.SobelFilterSwitchLabel.Position = [440 14 66 22];
            app.SobelFilterSwitchLabel.Text = 'Sobel Filter';

            % Create SobelFilterSwitch
            app.SobelFilterSwitch = uiswitch(app.MaskingTab, 'slider');
            app.SobelFilterSwitch.ValueChangedFcn = createCallbackFcn(app, @SobelFilterSwitchValueChanged, true);
            app.SobelFilterSwitch.Position = [531 14 45 20];

            % Create GoButton_8
            app.GoButton_8 = uibutton(app.MaskingTab, 'push');
            app.GoButton_8.ButtonPushedFcn = createCallbackFcn(app, @GoButton_8Pushed, true);
            app.GoButton_8.Position = [226 116 68 22];
            app.GoButton_8.Text = 'Go';

            % Create ResetButton_8
            app.ResetButton_8 = uibutton(app.MaskingTab, 'push');
            app.ResetButton_8.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_8Pushed, true);
            app.ResetButton_8.Position = [305 116 55 22];
            app.ResetButton_8.Text = 'Reset';

            % Create LaplacianFilteralphaLabel
            app.LaplacianFilteralphaLabel = uilabel(app.MaskingTab);
            app.LaplacianFilteralphaLabel.HorizontalAlignment = 'right';
            app.LaplacianFilteralphaLabel.Position = [12 116 120 22];
            app.LaplacianFilteralphaLabel.Text = 'Laplacian Filter alpha';

            % Create Lfa
            app.Lfa = uieditfield(app.MaskingTab, 'numeric');
            app.Lfa.ValueChangedFcn = createCallbackFcn(app, @LfaValueChanged, true);
            app.Lfa.Position = [138 116 34 22];

            % Create Label
            app.Label = uilabel(app.MaskingTab);
            app.Label.Position = [175 116 52 22];
            app.Label.Text = '[0, 1],0.2';

            % Create GoButton_9
            app.GoButton_9 = uibutton(app.MaskingTab, 'push');
            app.GoButton_9.ButtonPushedFcn = createCallbackFcn(app, @GoButton_9Pushed, true);
            app.GoButton_9.Position = [316 10 52 22];
            app.GoButton_9.Text = 'Go';

            % Create ResetButton_9
            app.ResetButton_9 = uibutton(app.MaskingTab, 'push');
            app.ResetButton_9.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_9Pushed, true);
            app.ResetButton_9.Position = [380 10 55 22];
            app.ResetButton_9.Text = 'Reset';

            % Create AmountLabel
            app.AmountLabel = uilabel(app.MaskingTab);
            app.AmountLabel.Position = [173 10 53 22];
            app.AmountLabel.Text = ', Amount';

            % Create Usfa
            app.Usfa = uieditfield(app.MaskingTab, 'numeric');
            app.Usfa.ValueChangedFcn = createCallbackFcn(app, @UsfaValueChanged, true);
            app.Usfa.Position = [227 10 33 22];

            % Create UnsharpFilterRaiusLabel
            app.UnsharpFilterRaiusLabel = uilabel(app.MaskingTab);
            app.UnsharpFilterRaiusLabel.HorizontalAlignment = 'right';
            app.UnsharpFilterRaiusLabel.Position = [13 10 115 22];
            app.UnsharpFilterRaiusLabel.Text = 'Unsharp Filter Raius';

            % Create Usfr
            app.Usfr = uieditfield(app.MaskingTab, 'numeric');
            app.Usfr.ValueChangedFcn = createCallbackFcn(app, @UsfrValueChanged, true);
            app.Usfr.Position = [134 10 34 22];

            % Create abcaxbandxcisdefaultvalueLabel
            app.abcaxbandxcisdefaultvalueLabel = uilabel(app.MaskingTab);
            app.abcaxbandxcisdefaultvalueLabel.FontWeight = 'bold';
            app.abcaxbandxcisdefaultvalueLabel.Position = [184 256 241 22];
            app.abcaxbandxcisdefaultvalueLabel.Text = ' [a, b], c  (a<x<b and x=c is default value)';

            % Create Label_2
            app.Label_2 = uilabel(app.MaskingTab);
            app.Label_2.Position = [262 10 52 22];
            app.Label_2.Text = '[0, 2],0.2';

            % Create PassfilterTab
            app.PassfilterTab = uitab(app.TabGroup);
            app.PassfilterTab.Title = 'Pass filter';

            % Create FrequencyDomainLabel
            app.FrequencyDomainLabel = uilabel(app.PassfilterTab);
            app.FrequencyDomainLabel.FontWeight = 'bold';
            app.FrequencyDomainLabel.Position = [232 277 114 22];
            app.FrequencyDomainLabel.Text = 'Frequency Domain';

            % Create GoButton_10
            app.GoButton_10 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_10.ButtonPushedFcn = createCallbackFcn(app, @GoButton_10Pushed, true);
            app.GoButton_10.Position = [287 240 68 22];
            app.GoButton_10.Text = 'Go';

            % Create ResetButton_10
            app.ResetButton_10 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_10.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_10Pushed, true);
            app.ResetButton_10.Position = [366 240 55 22];
            app.ResetButton_10.Text = 'Reset';

            % Create IdealHighPassfilterCutofffrequencyLabel
            app.IdealHighPassfilterCutofffrequencyLabel = uilabel(app.PassfilterTab);
            app.IdealHighPassfilterCutofffrequencyLabel.HorizontalAlignment = 'right';
            app.IdealHighPassfilterCutofffrequencyLabel.Position = [20 240 214 22];
            app.IdealHighPassfilterCutofffrequencyLabel.Text = 'Ideal High Pass filter, Cut off frequency';

            % Create Ihp
            app.Ihp = uieditfield(app.PassfilterTab, 'numeric');
            app.Ihp.ValueChangedFcn = createCallbackFcn(app, @IhpValueChanged, true);
            app.Ihp.Position = [240 240 34 22];

            % Create GoButton_11
            app.GoButton_11 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_11.ButtonPushedFcn = createCallbackFcn(app, @GoButton_11Pushed, true);
            app.GoButton_11.Position = [284 206 68 22];
            app.GoButton_11.Text = 'Go';

            % Create ResetButton_11
            app.ResetButton_11 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_11.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_11Pushed, true);
            app.ResetButton_11.Position = [363 206 55 22];
            app.ResetButton_11.Text = 'Reset';

            % Create IdealLowPassfilterCutofffrequencyLabel
            app.IdealLowPassfilterCutofffrequencyLabel = uilabel(app.PassfilterTab);
            app.IdealLowPassfilterCutofffrequencyLabel.HorizontalAlignment = 'right';
            app.IdealLowPassfilterCutofffrequencyLabel.Position = [20 206 211 22];
            app.IdealLowPassfilterCutofffrequencyLabel.Text = 'Ideal Low Pass filter, Cut off frequency';

            % Create Ilp
            app.Ilp = uieditfield(app.PassfilterTab, 'numeric');
            app.Ilp.ValueChangedFcn = createCallbackFcn(app, @IlpValueChanged, true);
            app.Ilp.Position = [237 206 34 22];

            % Create GoButton_12
            app.GoButton_12 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_12.ButtonPushedFcn = createCallbackFcn(app, @GoButton_12Pushed, true);
            app.GoButton_12.Position = [402 173 68 22];
            app.GoButton_12.Text = 'Go';

            % Create ResetButton_12
            app.ResetButton_12 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_12.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_12Pushed, true);
            app.ResetButton_12.Position = [481 173 55 22];
            app.ResetButton_12.Text = 'Reset';

            % Create ButterworthHighPassfilterCutofffrequencyLabel
            app.ButterworthHighPassfilterCutofffrequencyLabel = uilabel(app.PassfilterTab);
            app.ButterworthHighPassfilterCutofffrequencyLabel.HorizontalAlignment = 'right';
            app.ButterworthHighPassfilterCutofffrequencyLabel.Position = [19 173 250 22];
            app.ButterworthHighPassfilterCutofffrequencyLabel.Text = 'Butterworth High Pass filter, Cut off frequency';

            % Create Bwhf
            app.Bwhf = uieditfield(app.PassfilterTab, 'numeric');
            app.Bwhf.ValueChangedFcn = createCallbackFcn(app, @BwhfValueChanged, true);
            app.Bwhf.Position = [275 173 34 22];

            % Create orderLabel
            app.orderLabel = uilabel(app.PassfilterTab);
            app.orderLabel.HorizontalAlignment = 'right';
            app.orderLabel.Position = [308 173 40 22];
            app.orderLabel.Text = ', order';

            % Create Bwho
            app.Bwho = uieditfield(app.PassfilterTab, 'numeric');
            app.Bwho.ValueChangedFcn = createCallbackFcn(app, @BwhoValueChanged, true);
            app.Bwho.Position = [354 173 34 22];

            % Create GoButton_13
            app.GoButton_13 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_13.ButtonPushedFcn = createCallbackFcn(app, @GoButton_13Pushed, true);
            app.GoButton_13.Position = [400 142 68 22];
            app.GoButton_13.Text = 'Go';

            % Create ResetButton_13
            app.ResetButton_13 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_13.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_13Pushed, true);
            app.ResetButton_13.Position = [479 142 55 22];
            app.ResetButton_13.Text = 'Reset';

            % Create ButterworthLowPassfilterCutofffrequencyLabel
            app.ButterworthLowPassfilterCutofffrequencyLabel = uilabel(app.PassfilterTab);
            app.ButterworthLowPassfilterCutofffrequencyLabel.HorizontalAlignment = 'right';
            app.ButterworthLowPassfilterCutofffrequencyLabel.Position = [20 142 247 22];
            app.ButterworthLowPassfilterCutofffrequencyLabel.Text = 'Butterworth Low Pass filter, Cut off frequency';

            % Create Bwlf
            app.Bwlf = uieditfield(app.PassfilterTab, 'numeric');
            app.Bwlf.ValueChangedFcn = createCallbackFcn(app, @BwlfValueChanged, true);
            app.Bwlf.Position = [273 142 34 22];

            % Create Bwlo
            app.Bwlo = uieditfield(app.PassfilterTab, 'numeric');
            app.Bwlo.ValueChangedFcn = createCallbackFcn(app, @BwloValueChanged, true);
            app.Bwlo.Position = [352 142 34 22];

            % Create orderLabel_2
            app.orderLabel_2 = uilabel(app.PassfilterTab);
            app.orderLabel_2.HorizontalAlignment = 'right';
            app.orderLabel_2.Position = [306 142 40 22];
            app.orderLabel_2.Text = ', order';

            % Create GoButton_14
            app.GoButton_14 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_14.ButtonPushedFcn = createCallbackFcn(app, @GoButton_14Pushed, true);
            app.GoButton_14.Position = [339 112 68 22];
            app.GoButton_14.Text = 'Go';

            % Create ResetButton_14
            app.ResetButton_14 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_14.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_14Pushed, true);
            app.ResetButton_14.Position = [418 112 55 22];
            app.ResetButton_14.Text = 'Reset';

            % Create IdealBandpassfilterBandradiusLabel
            app.IdealBandpassfilterBandradiusLabel = uilabel(app.PassfilterTab);
            app.IdealBandpassfilterBandradiusLabel.HorizontalAlignment = 'right';
            app.IdealBandpassfilterBandradiusLabel.Position = [22 112 184 22];
            app.IdealBandpassfilterBandradiusLabel.Text = 'Ideal Bandpass filter, Band radius';

            % Create Ibpr
            app.Ibpr = uieditfield(app.PassfilterTab, 'numeric');
            app.Ibpr.ValueChangedFcn = createCallbackFcn(app, @IbprValueChanged, true);
            app.Ibpr.Position = [212 112 34 22];

            % Create Ibpw
            app.Ibpw = uieditfield(app.PassfilterTab, 'numeric');
            app.Ibpw.ValueChangedFcn = createCallbackFcn(app, @IbpwValueChanged, true);
            app.Ibpw.Position = [291 112 34 22];

            % Create widthLabel
            app.widthLabel = uilabel(app.PassfilterTab);
            app.widthLabel.HorizontalAlignment = 'right';
            app.widthLabel.Position = [245 112 40 22];
            app.widthLabel.Text = ', width';

            % Create GoButton_15
            app.GoButton_15 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_15.ButtonPushedFcn = createCallbackFcn(app, @GoButton_15Pushed, true);
            app.GoButton_15.Position = [343 80 68 22];
            app.GoButton_15.Text = 'Go';

            % Create ResetButton_15
            app.ResetButton_15 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_15.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_15Pushed, true);
            app.ResetButton_15.Position = [422 80 55 22];
            app.ResetButton_15.Text = 'Reset';

            % Create IdealBandrejectfilterBandradiusLabel
            app.IdealBandrejectfilterBandradiusLabel = uilabel(app.PassfilterTab);
            app.IdealBandrejectfilterBandradiusLabel.HorizontalAlignment = 'right';
            app.IdealBandrejectfilterBandradiusLabel.Position = [22 80 188 22];
            app.IdealBandrejectfilterBandradiusLabel.Text = 'Ideal Bandreject filter, Band radius';

            % Create Ibrr
            app.Ibrr = uieditfield(app.PassfilterTab, 'numeric');
            app.Ibrr.ValueChangedFcn = createCallbackFcn(app, @IbrrValueChanged, true);
            app.Ibrr.Position = [216 80 34 22];

            % Create Ibrw
            app.Ibrw = uieditfield(app.PassfilterTab, 'numeric');
            app.Ibrw.ValueChangedFcn = createCallbackFcn(app, @IbrwValueChanged, true);
            app.Ibrw.Position = [295 80 34 22];

            % Create widthLabel_2
            app.widthLabel_2 = uilabel(app.PassfilterTab);
            app.widthLabel_2.HorizontalAlignment = 'right';
            app.widthLabel_2.Position = [249 80 40 22];
            app.widthLabel_2.Text = ', width';

            % Create GoButton_16
            app.GoButton_16 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_16.ButtonPushedFcn = createCallbackFcn(app, @GoButton_16Pushed, true);
            app.GoButton_16.Position = [224 50 68 22];
            app.GoButton_16.Text = 'Go';

            % Create ResetButton_16
            app.ResetButton_16 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_16.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_16Pushed, true);
            app.ResetButton_16.Position = [303 50 55 22];
            app.ResetButton_16.Text = 'Reset';

            % Create NotchfilteringrowLabel
            app.NotchfilteringrowLabel = uilabel(app.PassfilterTab);
            app.NotchfilteringrowLabel.HorizontalAlignment = 'right';
            app.NotchfilteringrowLabel.Position = [22 50 108 22];
            app.NotchfilteringrowLabel.Text = 'Notch filtering, row ';

            % Create Nfr1
            app.Nfr1 = uieditfield(app.PassfilterTab, 'numeric');
            app.Nfr1.ValueChangedFcn = createCallbackFcn(app, @Nfr1ValueChanged, true);
            app.Nfr1.Position = [132 50 34 22];

            % Create NotchfilteringcolumnLabel
            app.NotchfilteringcolumnLabel = uilabel(app.PassfilterTab);
            app.NotchfilteringcolumnLabel.HorizontalAlignment = 'right';
            app.NotchfilteringcolumnLabel.Position = [21 22 128 22];
            app.NotchfilteringcolumnLabel.Text = 'Notch filtering, column ';

            % Create Nfc1
            app.Nfc1 = uieditfield(app.PassfilterTab, 'numeric');
            app.Nfc1.ValueChangedFcn = createCallbackFcn(app, @Nfc1ValueChanged, true);
            app.Nfc1.Position = [151 22 34 22];

            % Create GoButton_17
            app.GoButton_17 = uibutton(app.PassfilterTab, 'push');
            app.GoButton_17.ButtonPushedFcn = createCallbackFcn(app, @GoButton_17Pushed, true);
            app.GoButton_17.Position = [243 22 68 22];
            app.GoButton_17.Text = 'Go';

            % Create ResetButton_17
            app.ResetButton_17 = uibutton(app.PassfilterTab, 'push');
            app.ResetButton_17.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_17Pushed, true);
            app.ResetButton_17.Position = [322 22 55 22];
            app.ResetButton_17.Text = 'Reset';

            % Create Nfr2
            app.Nfr2 = uieditfield(app.PassfilterTab, 'numeric');
            app.Nfr2.ValueChangedFcn = createCallbackFcn(app, @Nfr2ValueChanged, true);
            app.Nfr2.Position = [180 50 34 22];

            % Create Nfc2
            app.Nfc2 = uieditfield(app.PassfilterTab, 'numeric');
            app.Nfc2.ValueChangedFcn = createCallbackFcn(app, @Nfc2ValueChanged, true);
            app.Nfc2.Position = [199 22 34 22];

            % Create Label_3
            app.Label_3 = uilabel(app.PassfilterTab);
            app.Label_3.Position = [171 50 25 22];
            app.Label_3.Text = ':';

            % Create Label_4
            app.Label_4 = uilabel(app.PassfilterTab);
            app.Label_4.Position = [190 22 25 22];
            app.Label_4.Text = ':';

            % Create NoiseTab
            app.NoiseTab = uitab(app.TabGroup);
            app.NoiseTab.Title = 'Noise';

            % Create GaussiannoiseSwitchLabel
            app.GaussiannoiseSwitchLabel = uilabel(app.NoiseTab);
            app.GaussiannoiseSwitchLabel.HorizontalAlignment = 'center';
            app.GaussiannoiseSwitchLabel.Position = [32 261 88 22];
            app.GaussiannoiseSwitchLabel.Text = 'Gaussian noise';

            % Create GaussiannoiseSwitch
            app.GaussiannoiseSwitch = uiswitch(app.NoiseTab, 'slider');
            app.GaussiannoiseSwitch.ValueChangedFcn = createCallbackFcn(app, @GaussiannoiseSwitchValueChanged, true);
            app.GaussiannoiseSwitch.Position = [173 262 45 20];

            % Create PoissonnoiseSwitchLabel
            app.PoissonnoiseSwitchLabel = uilabel(app.NoiseTab);
            app.PoissonnoiseSwitchLabel.HorizontalAlignment = 'center';
            app.PoissonnoiseSwitchLabel.Position = [33 220 80 22];
            app.PoissonnoiseSwitchLabel.Text = 'Poisson noise';

            % Create PoissonnoiseSwitch
            app.PoissonnoiseSwitch = uiswitch(app.NoiseTab, 'slider');
            app.PoissonnoiseSwitch.ValueChangedFcn = createCallbackFcn(app, @PoissonnoiseSwitchValueChanged, true);
            app.PoissonnoiseSwitch.Position = [172 220 45 20];

            % Create SaltPeppernoiseSwitchLabel
            app.SaltPeppernoiseSwitchLabel = uilabel(app.NoiseTab);
            app.SaltPeppernoiseSwitchLabel.HorizontalAlignment = 'center';
            app.SaltPeppernoiseSwitchLabel.Position = [31 181 112 22];
            app.SaltPeppernoiseSwitchLabel.Text = 'Salt & Pepper noise';

            % Create SaltPeppernoiseSwitch
            app.SaltPeppernoiseSwitch = uiswitch(app.NoiseTab, 'slider');
            app.SaltPeppernoiseSwitch.ValueChangedFcn = createCallbackFcn(app, @SaltPeppernoiseSwitchValueChanged, true);
            app.SaltPeppernoiseSwitch.Position = [172 181 45 20];

            % Create SpecklenoiseSwitchLabel
            app.SpecklenoiseSwitchLabel = uilabel(app.NoiseTab);
            app.SpecklenoiseSwitchLabel.HorizontalAlignment = 'center';
            app.SpecklenoiseSwitchLabel.Position = [32 143 80 22];
            app.SpecklenoiseSwitchLabel.Text = 'Speckle noise';

            % Create SpecklenoiseSwitch
            app.SpecklenoiseSwitch = uiswitch(app.NoiseTab, 'slider');
            app.SpecklenoiseSwitch.ValueChangedFcn = createCallbackFcn(app, @SpecklenoiseSwitchValueChanged, true);
            app.SpecklenoiseSwitch.Position = [172 143 45 20];

            % Create RestorationTab
            app.RestorationTab = uitab(app.TabGroup);
            app.RestorationTab.Title = 'Restoration';

            % Create XLabel_4
            app.XLabel_4 = uilabel(app.RestorationTab);
            app.XLabel_4.Position = [185 256 10 22];
            app.XLabel_4.Text = 'X';

            % Create armn
            app.armn = uieditfield(app.RestorationTab, 'numeric');
            app.armn.ValueChangedFcn = createCallbackFcn(app, @armnValueChanged, true);
            app.armn.Position = [197 256 33 22];

            % Create GoButton_18
            app.GoButton_18 = uibutton(app.RestorationTab, 'push');
            app.GoButton_18.ButtonPushedFcn = createCallbackFcn(app, @GoButton_18Pushed, true);
            app.GoButton_18.Position = [243 256 68 22];
            app.GoButton_18.Text = 'Go';

            % Create ResetButton_18
            app.ResetButton_18 = uibutton(app.RestorationTab, 'push');
            app.ResetButton_18.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_18Pushed, true);
            app.ResetButton_18.Position = [322 256 55 22];
            app.ResetButton_18.Text = 'Reset';

            % Create ArithmeticmeanmxnLabel
            app.ArithmeticmeanmxnLabel = uilabel(app.RestorationTab);
            app.ArithmeticmeanmxnLabel.HorizontalAlignment = 'right';
            app.ArithmeticmeanmxnLabel.Position = [15 256 126 22];
            app.ArithmeticmeanmxnLabel.Text = 'Arithmetic mean (mxn)';

            % Create armm
            app.armm = uieditfield(app.RestorationTab, 'numeric');
            app.armm.ValueChangedFcn = createCallbackFcn(app, @armmValueChanged, true);
            app.armm.Position = [147 256 34 22];

            % Create XLabel_5
            app.XLabel_5 = uilabel(app.RestorationTab);
            app.XLabel_5.Position = [181 223 10 22];
            app.XLabel_5.Text = 'X';

            % Create hmn
            app.hmn = uieditfield(app.RestorationTab, 'numeric');
            app.hmn.ValueChangedFcn = createCallbackFcn(app, @hmnValueChanged, true);
            app.hmn.Position = [193 223 33 22];

            % Create GoButton_19
            app.GoButton_19 = uibutton(app.RestorationTab, 'push');
            app.GoButton_19.ButtonPushedFcn = createCallbackFcn(app, @GoButton_19Pushed, true);
            app.GoButton_19.Position = [239 223 68 22];
            app.GoButton_19.Text = 'Go';

            % Create ResetButton_19
            app.ResetButton_19 = uibutton(app.RestorationTab, 'push');
            app.ResetButton_19.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_19Pushed, true);
            app.ResetButton_19.Position = [318 223 55 22];
            app.ResetButton_19.Text = 'Reset';

            % Create HarmonicmeanmxnLabel
            app.HarmonicmeanmxnLabel = uilabel(app.RestorationTab);
            app.HarmonicmeanmxnLabel.HorizontalAlignment = 'right';
            app.HarmonicmeanmxnLabel.Position = [13 223 124 22];
            app.HarmonicmeanmxnLabel.Text = 'Harmonic mean (mxn)';

            % Create hmm
            app.hmm = uieditfield(app.RestorationTab, 'numeric');
            app.hmm.ValueChangedFcn = createCallbackFcn(app, @hmmValueChanged, true);
            app.hmm.Position = [143 223 34 22];

            % Create XLabel_7
            app.XLabel_7 = uilabel(app.RestorationTab);
            app.XLabel_7.Position = [161 191 10 22];
            app.XLabel_7.Text = 'X';

            % Create mfn
            app.mfn = uieditfield(app.RestorationTab, 'numeric');
            app.mfn.ValueChangedFcn = createCallbackFcn(app, @mfnValueChanged, true);
            app.mfn.Position = [173 191 33 22];

            % Create GoButton_21
            app.GoButton_21 = uibutton(app.RestorationTab, 'push');
            app.GoButton_21.ButtonPushedFcn = createCallbackFcn(app, @GoButton_21Pushed, true);
            app.GoButton_21.Position = [219 191 68 22];
            app.GoButton_21.Text = 'Go';

            % Create ResetButton_21
            app.ResetButton_21 = uibutton(app.RestorationTab, 'push');
            app.ResetButton_21.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_21Pushed, true);
            app.ResetButton_21.Position = [298 191 55 22];
            app.ResetButton_21.Text = 'Reset';

            % Create MedianfiltermxnLabel
            app.MedianfiltermxnLabel = uilabel(app.RestorationTab);
            app.MedianfiltermxnLabel.HorizontalAlignment = 'right';
            app.MedianfiltermxnLabel.Position = [12 191 105 22];
            app.MedianfiltermxnLabel.Text = 'Median filter (mxn)';

            % Create mfm
            app.mfm = uieditfield(app.RestorationTab, 'numeric');
            app.mfm.ValueChangedFcn = createCallbackFcn(app, @mfmValueChanged, true);
            app.mfm.Position = [123 191 34 22];

            % Create GoButton_22
            app.GoButton_22 = uibutton(app.RestorationTab, 'push');
            app.GoButton_22.ButtonPushedFcn = createCallbackFcn(app, @GoButton_22Pushed, true);
            app.GoButton_22.Position = [165 161 68 22];
            app.GoButton_22.Text = 'Go';

            % Create ResetButton_22
            app.ResetButton_22 = uibutton(app.RestorationTab, 'push');
            app.ResetButton_22.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_22Pushed, true);
            app.ResetButton_22.Position = [244 161 55 22];
            app.ResetButton_22.Text = 'Reset';

            % Create Maxfiltersize3Label
            app.Maxfiltersize3Label = uilabel(app.RestorationTab);
            app.Maxfiltersize3Label.HorizontalAlignment = 'right';
            app.Maxfiltersize3Label.Position = [12 161 104 22];
            app.Maxfiltersize3Label.Text = 'Max filter size (>3)';

            % Create mxf
            app.mxf = uieditfield(app.RestorationTab, 'numeric');
            app.mxf.ValueChangedFcn = createCallbackFcn(app, @mxfValueChanged, true);
            app.mxf.Position = [122 161 34 22];

            % Create GoButton_23
            app.GoButton_23 = uibutton(app.RestorationTab, 'push');
            app.GoButton_23.ButtonPushedFcn = createCallbackFcn(app, @GoButton_23Pushed, true);
            app.GoButton_23.Position = [165 132 68 22];
            app.GoButton_23.Text = 'Go';

            % Create ResetButton_23
            app.ResetButton_23 = uibutton(app.RestorationTab, 'push');
            app.ResetButton_23.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_23Pushed, true);
            app.ResetButton_23.Position = [244 132 55 22];
            app.ResetButton_23.Text = 'Reset';

            % Create Minfiltersize3Label
            app.Minfiltersize3Label = uilabel(app.RestorationTab);
            app.Minfiltersize3Label.HorizontalAlignment = 'right';
            app.Minfiltersize3Label.Position = [12 132 101 22];
            app.Minfiltersize3Label.Text = 'Min filter size (>3)';

            % Create mnf
            app.mnf = uieditfield(app.RestorationTab, 'numeric');
            app.mnf.ValueChangedFcn = createCallbackFcn(app, @mnfValueChanged, true);
            app.mnf.Position = [119 132 34 22];

            % Create WeinerfilterSwitchLabel
            app.WeinerfilterSwitchLabel = uilabel(app.RestorationTab);
            app.WeinerfilterSwitchLabel.HorizontalAlignment = 'center';
            app.WeinerfilterSwitchLabel.Position = [16 65 69 22];
            app.WeinerfilterSwitchLabel.Text = 'Weiner filter';

            % Create WeinerfilterSwitch
            app.WeinerfilterSwitch = uiswitch(app.RestorationTab, 'slider');
            app.WeinerfilterSwitch.ValueChangedFcn = createCallbackFcn(app, @WeinerfilterSwitchValueChanged, true);
            app.WeinerfilterSwitch.Position = [147 66 45 20];

            % Create IterativeNonLinearRestorationIterationSwitchLabel
            app.IterativeNonLinearRestorationIterationSwitchLabel = uilabel(app.RestorationTab);
            app.IterativeNonLinearRestorationIterationSwitchLabel.HorizontalAlignment = 'center';
            app.IterativeNonLinearRestorationIterationSwitchLabel.Position = [15 40 241 22];
            app.IterativeNonLinearRestorationIterationSwitchLabel.Text = 'Iterative Non-Linear Restoration, Iteration = ';

            % Create IterativeNonLinearRestorationIterationSwitch
            app.IterativeNonLinearRestorationIterationSwitch = uiswitch(app.RestorationTab, 'slider');
            app.IterativeNonLinearRestorationIterationSwitch.ValueChangedFcn = createCallbackFcn(app, @IterativeNonLinearRestorationIterationSwitchValueChanged, true);
            app.IterativeNonLinearRestorationIterationSwitch.Position = [322 41 45 20];

            % Create iteration
            app.iteration = uieditfield(app.RestorationTab, 'numeric');
            app.iteration.ValueChangedFcn = createCallbackFcn(app, @iterationValueChanged, true);
            app.iteration.Position = [254 40 34 22];

            % Create ReconstructionTab
            app.ReconstructionTab = uitab(app.TabGroup);
            app.ReconstructionTab.Title = 'Reconstruction';

            % Create RT
            app.RT = uiaxes(app.ReconstructionTab);
            app.RT.XTick = [];
            app.RT.XTickLabel = {'[ ]'};
            app.RT.YTick = [];
            app.RT.Position = [23 9 302 183];

            % Create ResetButton_24
            app.ResetButton_24 = uibutton(app.ReconstructionTab, 'push');
            app.ResetButton_24.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_24Pushed, true);
            app.ResetButton_24.Position = [508 201 55 56];
            app.ResetButton_24.Text = 'Reset';

            % Create GoButton_24
            app.GoButton_24 = uibutton(app.ReconstructionTab, 'push');
            app.GoButton_24.ButtonPushedFcn = createCallbackFcn(app, @GoButton_24Pushed, true);
            app.GoButton_24.Position = [429 201 68 56];
            app.GoButton_24.Text = 'Go';

            % Create irtn
            app.irtn = uieditfield(app.ReconstructionTab, 'numeric');
            app.irtn.ValueChangedFcn = createCallbackFcn(app, @irtnValueChanged, true);
            app.irtn.Position = [214 202 36 22];

            % Create ProjectionthetaEditFieldLabel_3
            app.ProjectionthetaEditFieldLabel_3 = uilabel(app.ReconstructionTab);
            app.ProjectionthetaEditFieldLabel_3.HorizontalAlignment = 'right';
            app.ProjectionthetaEditFieldLabel_3.FontWeight = 'bold';
            app.ProjectionthetaEditFieldLabel_3.Position = [198 202 10 22];
            app.ProjectionthetaEditFieldLabel_3.Text = ':';

            % Create rtn
            app.rtn = uieditfield(app.ReconstructionTab, 'numeric');
            app.rtn.ValueChangedFcn = createCallbackFcn(app, @rtnValueChanged, true);
            app.rtn.Position = [184 235 36 22];

            % Create ProjectionthetaEditFieldLabel_2
            app.ProjectionthetaEditFieldLabel_2 = uilabel(app.ReconstructionTab);
            app.ProjectionthetaEditFieldLabel_2.HorizontalAlignment = 'right';
            app.ProjectionthetaEditFieldLabel_2.FontWeight = 'bold';
            app.ProjectionthetaEditFieldLabel_2.Position = [168 235 10 22];
            app.ProjectionthetaEditFieldLabel_2.Text = ':';

            % Create RadonTransformLabel
            app.RadonTransformLabel = uilabel(app.ReconstructionTab);
            app.RadonTransformLabel.FontSize = 15;
            app.RadonTransformLabel.FontWeight = 'bold';
            app.RadonTransformLabel.Position = [215 273 130 22];
            app.RadonTransformLabel.Text = 'Radon Transform';

            % Create rtm
            app.rtm = uieditfield(app.ReconstructionTab, 'numeric');
            app.rtm.ValueChangedFcn = createCallbackFcn(app, @rtmValueChanged, true);
            app.rtm.Position = [133 235 36 22];

            % Create Projectiontheta0EditFieldLabel
            app.Projectiontheta0EditFieldLabel = uilabel(app.ReconstructionTab);
            app.Projectiontheta0EditFieldLabel.HorizontalAlignment = 'right';
            app.Projectiontheta0EditFieldLabel.Position = [21 235 106 22];
            app.Projectiontheta0EditFieldLabel.Text = 'Projection theta 0 :';

            % Create irtm
            app.irtm = uieditfield(app.ReconstructionTab, 'numeric');
            app.irtm.ValueChangedFcn = createCallbackFcn(app, @irtmValueChanged, true);
            app.irtm.Position = [163 202 36 22];

            % Create BackProjectiontheta0EditFieldLabel
            app.BackProjectiontheta0EditFieldLabel = uilabel(app.ReconstructionTab);
            app.BackProjectiontheta0EditFieldLabel.HorizontalAlignment = 'right';
            app.BackProjectiontheta0EditFieldLabel.Position = [21 202 136 22];
            app.BackProjectiontheta0EditFieldLabel.Text = 'Back Projection theta 0 :';

            % Create InterpolationDropDownLabel
            app.InterpolationDropDownLabel = uilabel(app.ReconstructionTab);
            app.InterpolationDropDownLabel.HorizontalAlignment = 'right';
            app.InterpolationDropDownLabel.Position = [248 232 78 22];
            app.InterpolationDropDownLabel.Text = ', Interpolation';

            % Create InterpolationDropDown
            app.InterpolationDropDown = uidropdown(app.ReconstructionTab);
            app.InterpolationDropDown.Items = {'linear', 'nearest', 'cubic', 'spline'};
            app.InterpolationDropDown.ValueChangedFcn = createCallbackFcn(app, @InterpolationDropDownValueChanged, true);
            app.InterpolationDropDown.Position = [341 232 84 22];
            app.InterpolationDropDown.Value = 'linear';

            % Create FilterDropDownLabel
            app.FilterDropDownLabel = uilabel(app.ReconstructionTab);
            app.FilterDropDownLabel.HorizontalAlignment = 'right';
            app.FilterDropDownLabel.Position = [287 200 39 22];
            app.FilterDropDownLabel.Text = ', Filter';

            % Create FilterDropDown
            app.FilterDropDown = uidropdown(app.ReconstructionTab);
            app.FilterDropDown.Items = {'None', 'Ram-Lak', 'Shepp-Logan', 'Cosine', 'Hamming', 'Hann'};
            app.FilterDropDown.ValueChangedFcn = createCallbackFcn(app, @FilterDropDownValueChanged, true);
            app.FilterDropDown.Position = [341 200 84 22];
            app.FilterDropDown.Value = 'None';

            % Create AffinetransformTab
            app.AffinetransformTab = uitab(app.TabGroup);
            app.AffinetransformTab.Title = 'Affine transform';

            % Create Label_5
            app.Label_5 = uilabel(app.AffinetransformTab);
            app.Label_5.Position = [195 239 25 22];
            app.Label_5.Text = ',';

            % Create scalingy
            app.scalingy = uieditfield(app.AffinetransformTab, 'numeric');
            app.scalingy.ValueChangedFcn = createCallbackFcn(app, @scalingyValueChanged, true);
            app.scalingy.Position = [207 239 33 22];

            % Create GoButton_25
            app.GoButton_25 = uibutton(app.AffinetransformTab, 'push');
            app.GoButton_25.ButtonPushedFcn = createCallbackFcn(app, @GoButton_25Pushed, true);
            app.GoButton_25.Position = [253 239 68 22];
            app.GoButton_25.Text = 'Go';

            % Create ScalingSxSyLabel
            app.ScalingSxSyLabel = uilabel(app.AffinetransformTab);
            app.ScalingSxSyLabel.HorizontalAlignment = 'right';
            app.ScalingSxSyLabel.Position = [57 239 93 22];
            app.ScalingSxSyLabel.Text = 'Scaling Sx, Sy : ';

            % Create scalingx
            app.scalingx = uieditfield(app.AffinetransformTab, 'numeric');
            app.scalingx.ValueChangedFcn = createCallbackFcn(app, @scalingxValueChanged, true);
            app.scalingx.Position = [156 239 34 22];

            % Create GoButton_26
            app.GoButton_26 = uibutton(app.AffinetransformTab, 'push');
            app.GoButton_26.ButtonPushedFcn = createCallbackFcn(app, @GoButton_26Pushed, true);
            app.GoButton_26.Position = [253 206 68 22];
            app.GoButton_26.Text = 'Go';

            % Create RotaionthetaLabel
            app.RotaionthetaLabel = uilabel(app.AffinetransformTab);
            app.RotaionthetaLabel.HorizontalAlignment = 'right';
            app.RotaionthetaLabel.Position = [70 206 77 22];
            app.RotaionthetaLabel.Text = 'Rotaion theta';

            % Create rotation
            app.rotation = uieditfield(app.AffinetransformTab, 'numeric');
            app.rotation.ValueChangedFcn = createCallbackFcn(app, @rotationValueChanged, true);
            app.rotation.Position = [156 206 34 22];

            % Create GoButton_27
            app.GoButton_27 = uibutton(app.AffinetransformTab, 'push');
            app.GoButton_27.ButtonPushedFcn = createCallbackFcn(app, @GoButton_27Pushed, true);
            app.GoButton_27.Position = [253 171 68 22];
            app.GoButton_27.Text = 'Go';

            % Create ShearHorizontalslopeLabel
            app.ShearHorizontalslopeLabel = uilabel(app.AffinetransformTab);
            app.ShearHorizontalslopeLabel.HorizontalAlignment = 'right';
            app.ShearHorizontalslopeLabel.Position = [19 171 127 22];
            app.ShearHorizontalslopeLabel.Text = 'Shear Horizontal slope';

            % Create shearh
            app.shearh = uieditfield(app.AffinetransformTab, 'numeric');
            app.shearh.ValueChangedFcn = createCallbackFcn(app, @shearhValueChanged, true);
            app.shearh.Position = [156 171 34 22];

            % Create GoButton_28
            app.GoButton_28 = uibutton(app.AffinetransformTab, 'push');
            app.GoButton_28.ButtonPushedFcn = createCallbackFcn(app, @GoButton_28Pushed, true);
            app.GoButton_28.Position = [253 139 68 22];
            app.GoButton_28.Text = 'Go';

            % Create ShearverticalslopeLabel
            app.ShearverticalslopeLabel = uilabel(app.AffinetransformTab);
            app.ShearverticalslopeLabel.HorizontalAlignment = 'right';
            app.ShearverticalslopeLabel.Position = [35 139 111 22];
            app.ShearverticalslopeLabel.Text = 'Shear vertical slope';

            % Create shearv
            app.shearv = uieditfield(app.AffinetransformTab, 'numeric');
            app.shearv.ValueChangedFcn = createCallbackFcn(app, @shearvValueChanged, true);
            app.shearv.Position = [156 139 34 22];

            % Create Label_6
            app.Label_6 = uilabel(app.AffinetransformTab);
            app.Label_6.Position = [194 106 25 22];
            app.Label_6.Text = ',';

            % Create dy
            app.dy = uieditfield(app.AffinetransformTab, 'numeric');
            app.dy.ValueChangedFcn = createCallbackFcn(app, @dyValueChanged, true);
            app.dy.Position = [206 106 33 22];

            % Create GoButton_29
            app.GoButton_29 = uibutton(app.AffinetransformTab, 'push');
            app.GoButton_29.ButtonPushedFcn = createCallbackFcn(app, @GoButton_29Pushed, true);
            app.GoButton_29.Position = [253 106 68 22];
            app.GoButton_29.Text = 'Go';

            % Create TranslationdxdyLabel
            app.TranslationdxdyLabel = uilabel(app.AffinetransformTab);
            app.TranslationdxdyLabel.HorizontalAlignment = 'right';
            app.TranslationdxdyLabel.Position = [40 106 110 22];
            app.TranslationdxdyLabel.Text = 'Translation dx, dy : ';

            % Create dx
            app.dx = uieditfield(app.AffinetransformTab, 'numeric');
            app.dx.ValueChangedFcn = createCallbackFcn(app, @dxValueChanged, true);
            app.dx.Position = [156 106 34 22];

            % Create GoButton_30
            app.GoButton_30 = uibutton(app.AffinetransformTab, 'push');
            app.GoButton_30.ButtonPushedFcn = createCallbackFcn(app, @GoButton_30Pushed, true);
            app.GoButton_30.Position = [253 71 68 22];
            app.GoButton_30.Text = 'Go';

            % Create ReflectionLabel
            app.ReflectionLabel = uilabel(app.AffinetransformTab);
            app.ReflectionLabel.HorizontalAlignment = 'right';
            app.ReflectionLabel.Position = [87 74 59 22];
            app.ReflectionLabel.Text = 'Reflection';

            % Create ResetButton_25
            app.ResetButton_25 = uibutton(app.AffinetransformTab, 'push');
            app.ResetButton_25.ButtonPushedFcn = createCallbackFcn(app, @ResetButton_25Pushed, true);
            app.ResetButton_25.Position = [378 140 55 56];
            app.ResetButton_25.Text = 'Reset';

            % Create AHETab
            app.AHETab = uitab(app.TabGroup);
            app.AHETab.Title = 'AHE';

            % Create AdapticeHistogramEqualizationLabel
            app.AdapticeHistogramEqualizationLabel = uilabel(app.AHETab);
            app.AdapticeHistogramEqualizationLabel.FontWeight = 'bold';
            app.AdapticeHistogramEqualizationLabel.Position = [164 273 194 22];
            app.AdapticeHistogramEqualizationLabel.Text = 'Adaptice Histogram Equalization';

            % Create ClipLimitLabel
            app.ClipLimitLabel = uilabel(app.AHETab);
            app.ClipLimitLabel.HorizontalAlignment = 'right';
            app.ClipLimitLabel.Position = [39 224 52 22];
            app.ClipLimitLabel.Text = 'ClipLimit';

            % Create cl
            app.cl = uieditfield(app.AHETab, 'numeric');
            app.cl.ValueChangedFcn = createCallbackFcn(app, @clValueChanged, true);
            app.cl.Position = [97 224 34 22];

            % Create Default001Label
            app.Default001Label = uilabel(app.AHETab);
            app.Default001Label.HorizontalAlignment = 'right';
            app.Default001Label.Position = [130 224 127 22];
            app.Default001Label.Text = '[ 0  1 ] , Default  : 0.01 ';

            % Create RangeDropDownLabel
            app.RangeDropDownLabel = uilabel(app.AHETab);
            app.RangeDropDownLabel.HorizontalAlignment = 'right';
            app.RangeDropDownLabel.Position = [40 191 41 22];
            app.RangeDropDownLabel.Text = 'Range';

            % Create RangeDropDown
            app.RangeDropDown = uidropdown(app.AHETab);
            app.RangeDropDown.Items = {'full', 'original'};
            app.RangeDropDown.ValueChangedFcn = createCallbackFcn(app, @RangeDropDownValueChanged, true);
            app.RangeDropDown.Position = [126 191 84 22];
            app.RangeDropDown.Value = 'full';

            % Create DistributionDropDownLabel
            app.DistributionDropDownLabel = uilabel(app.AHETab);
            app.DistributionDropDownLabel.HorizontalAlignment = 'right';
            app.DistributionDropDownLabel.Position = [39 160 66 22];
            app.DistributionDropDownLabel.Text = 'Distribution';

            % Create DistributionDropDown
            app.DistributionDropDown = uidropdown(app.AHETab);
            app.DistributionDropDown.Items = {'uniform', 'rayleigh', 'exponential'};
            app.DistributionDropDown.ValueChangedFcn = createCallbackFcn(app, @DistributionDropDownValueChanged, true);
            app.DistributionDropDown.Position = [126 159 84 22];
            app.DistributionDropDown.Value = 'uniform';

            % Create ahe
            app.ahe = uiswitch(app.AHETab, 'slider');
            app.ahe.ValueChangedFcn = createCallbackFcn(app, @aheValueChanged, true);
            app.ahe.Position = [296 191 45 20];

            % Create AllResetButton
            app.AllResetButton = uibutton(app.UIFigure, 'push');
            app.AllResetButton.ButtonPushedFcn = createCallbackFcn(app, @AllResetButtonPushed, true);
            app.AllResetButton.FontWeight = 'bold';
            app.AllResetButton.Position = [1133 317 100 22];
            app.AllResetButton.Text = 'All Reset';

            % Create HistogramLabel
            app.HistogramLabel = uilabel(app.UIFigure);
            app.HistogramLabel.FontSize = 15;
            app.HistogramLabel.FontWeight = 'bold';
            app.HistogramLabel.Position = [905 255 80 22];
            app.HistogramLabel.Text = 'Histogram';

            % Create gamma1Button
            app.gamma1Button = uibutton(app.UIFigure, 'push');
            app.gamma1Button.ButtonPushedFcn = createCallbackFcn(app, @gamma1ButtonPushed, true);
            app.gamma1Button.Position = [795 535 20 21];
            app.gamma1Button.Text = '1';

            % Create tgv
            app.tgv = uieditfield(app.UIFigure, 'numeric');
            app.tgv.ValueChangedFcn = createCallbackFcn(app, @tgvValueChanged, true);
            app.tgv.Position = [762 333 34 21];

            % Create gamma1Button_2
            app.gamma1Button_2 = uibutton(app.UIFigure, 'push');
            app.gamma1Button_2.ButtonPushedFcn = createCallbackFcn(app, @gamma1Button_2Pushed, true);
            app.gamma1Button_2.Position = [795 634 32 22];
            app.gamma1Button_2.Text = '1.5';

            % Create gamma1Button_3
            app.gamma1Button_3 = uibutton(app.UIFigure, 'push');
            app.gamma1Button_3.ButtonPushedFcn = createCallbackFcn(app, @gamma1Button_3Pushed, true);
            app.gamma1Button_3.Position = [795 443 32 22];
            app.gamma1Button_3.Text = '0.5';

            % Create gamma1Button_4
            app.gamma1Button_4 = uibutton(app.UIFigure, 'push');
            app.gamma1Button_4.ButtonPushedFcn = createCallbackFcn(app, @gamma1Button_4Pushed, true);
            app.gamma1Button_4.Position = [795 715 25 22];
            app.gamma1Button_4.Text = '2';

            % Create gamma1Button_5
            app.gamma1Button_5 = uibutton(app.UIFigure, 'push');
            app.gamma1Button_5.ButtonPushedFcn = createCallbackFcn(app, @gamma1Button_5Pushed, true);
            app.gamma1Button_5.Position = [795 355 25 22];
            app.gamma1Button_5.Text = '0';

            % Create GrayscaleSwitchLabel
            app.GrayscaleSwitchLabel = uilabel(app.UIFigure);
            app.GrayscaleSwitchLabel.HorizontalAlignment = 'center';
            app.GrayscaleSwitchLabel.FontWeight = 'bold';
            app.GrayscaleSwitchLabel.Position = [1459 291 63 22];
            app.GrayscaleSwitchLabel.Text = 'Grayscale';

            % Create GrayscaleSwitch
            app.GrayscaleSwitch = uiswitch(app.UIFigure, 'slider');
            app.GrayscaleSwitch.ValueChangedFcn = createCallbackFcn(app, @GrayscaleSwitchValueChanged, true);
            app.GrayscaleSwitch.Position = [1465 311 45 20];

            % Create DFTSwitchLabel_2
            app.DFTSwitchLabel_2 = uilabel(app.UIFigure);
            app.DFTSwitchLabel_2.HorizontalAlignment = 'center';
            app.DFTSwitchLabel_2.FontSize = 14;
            app.DFTSwitchLabel_2.FontWeight = 'bold';
            app.DFTSwitchLabel_2.Position = [1360 285 33 22];
            app.DFTSwitchLabel_2.Text = 'DFT';

            % Create SaveasButton
            app.SaveasButton = uibutton(app.UIFigure, 'push');
            app.SaveasButton.ButtonPushedFcn = createCallbackFcn(app, @SaveasButtonPushed, true);
            app.SaveasButton.FontWeight = 'bold';
            app.SaveasButton.Position = [1240 317 100 22];
            app.SaveasButton.Text = 'Save as';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Image_Processing_App_for_Thesis_Reference_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end