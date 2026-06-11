function[eig_lambda,eig_z,cProny,yFit,yData,Ts,Fs,resid,MSE,MRelAbsErr]=...
		Prony_Fit(yMeas,n,Ts,dec,method,pl)
	% fit the data using damped complex exponentials series
	%
	% Fiting model:
	%			yEst(k)=cProny*exp(eig_lambda*Ts*k)
	%			yEst(k)=cProny*z^k
	%
	%__________________________________________________________________________________________________
	% Refrences:
	%
	%
	%
	%
	%__________________________________________________________________________________________________
	%
	% Function INPUTS:
	%			yMeas	- measured data as a collumn vector yMeas(:,nData)
	%			n		- system order used to fit the data
	%			Ts		- Sample time of the timeseries
	%			dec		- decimation (use every 'dec' sample), default = 1
	%			method	- string specifiying the method of fitting:
	%								- 'polly'	- clasicall prony method using two linear AR fits
	%											- unstable for large ammount of data points
	%								- 'pencil'	- Matrix pencil method, more stable
	%								- default is 'pencil'
	%			pl	- 1 - plot the system response and poles
	%					- 0 - do not plot. 
	%
	% Function OUTPUTS: 
	%			eig_lambda	- Continous fitted system eigenvalues 
	%			eig_z		- Discrete fitted system eigenvalues 
	%			cProny		- Fiting complex exponential coefficients 
	%			yFit		- Estimated data at kTs time points
	%			yData		- Decimated measured data
	%			[Ts,Fs]		- Decimated sampling time and sampling frequency		
	%			resid		- Residual between estimated and measured data
	%			MSE			- Mean-squared error
	%			MRelAbsErr 	- Mean-relative-absolute-error
	%
	% Created by: Tin Bensic, 2019.
	%				tin.bensic@ferit.hr
	%				Faculty of electrical engineering, computer science and information technology
	%				J. J. Strossmayer University Osijek
	%				Kneza Trpimira 2B, 31000 Osijek
	%**************************************************************************************************											
	%% Setting default values for function arguments:
	%**************************************************************************************************		
	
	if nargin < 5
		pl = true;
	end
	
	if nargin < 4
		method = 'pencil';
		pl = true;
	end
	
	if nargin < 3
		method = 'pencil';
		dec = 1;
		pl = true;
	end
			
	%**************************************************************************************************											
	%% Data decimation and formating
	%**************************************************************************************************
			
	Ts = dec * Ts;
	Fs = 1 / Ts;
	yData = yMeas(1:dec:end);
	
	%**************************************************************************************************											
	%% Fiting the data:
	%**************************************************************************************************											

	% Step 1: Computing the eigenvalues:
	
	switch method
		
		case 'polly'
			
			yData = [zeros(n,1); yData]; % zero padding the data
					
			Phi =  -toeplitz(yData(n : end), fliplr(yData(1 : n).')); % Create Toeplitz form matrix 'Phi' from data points
			
			
			% The equation:	 yData = Phi * Theta
			%
			% Solved by least squares aproximation using Moore-Penrose pseudoinverse			
			%
			%			theta = (Phi' * Phi)^{-1} * Phi' * yData
			%
			% Matlab implemented using QR factorisation with operator '\'
			yData = yData(n:end);
			theta = Phi \ yData; % Solve the least squares
			
			den = [1;  theta].';
			
			eig_z = roots(den); % Compute the discrete eigenvalues
			
		case 'pencil'
			
			try
			Phi = hankel(yData(1 : end - n), yData(end - n : end)); % Form the Henkel matrix 'Phi' from data points
			catch 
				disp(size(yData));
				error('Greska');
			end
			% separate Henkel matrix into two matrices that form a matrix pencil
			Y1 = Phi(:,1:end-1);		

			Y2 = Phi(:,2:end);
								
			eig_z = eig(pinv(Y1) * Y2);  % Compute the discrete eigenvalues using Singular value decomposition

	end
			
	%__________________________________________________________________________________________________________________
	% Step 2: Compute the Prony coefficients 	
	

	Z = zeros(length(yData),n); % create matrix Z from powers of discrete eignevalues 

	for l= 1 : n
		Z(:,l)=eig_z(l).^(0:length(yData)-1);
	end
	
	
warning('off','all');

	cProny = Z \ yData; % Compute the prony coefficients using least squares aproximation
		
warning('on','all');


	%__________________________________________________________________________________________________________________
	% Step 3: Compute the continous system eigenvalues and estimated system response
	
	eig_lambda = log(eig_z) * Fs; % Continous time eigienvalues
	
	yFit = real(Z*cProny); % Estimated system response, real - needed to convert data (imag part is ^-20).
	
	resid = yData - yFit;
	
 	MSE = 1/length(yData) * sum(resid.^2) ;
	MRelAbsErr = 1/length(yData) * (sum(abs(resid)));

% 	MSE = sum(abs(resid));
	
	
	%**************************************************************************************************											
	%% PLOTING The results
	%**************************************************************************************************											

	if pl
		time = 0:Ts:Ts*(length(yData)-1);
		time = time.';
		
		a = figure;
		a.Name = 'Prony_Fitting_result';
		subplot(2,1,1);
		p = plot(time,[yData yFit]);
		ax = gca;
		p(1).LineWidth = 1;
		p(1).LineStyle = ':';
		p(1).Color = 'k';
		p(2).LineWidth = 0.5;
		p(2).LineStyle = '-';
		p(2).Color = 'k';
				
		legend('Measured Data', 'Estimated Data');
		title('Damped complex exponential fit to data');
		ax.XLabel.String = 'time [s]';
		ax.YLabel.String = 'amplitude';
		
		
		subplot(2,1,2);
		p = plot(time,resid);
		ax = gca;
		p.Color = 'k';
		p.LineWidth = 1;
		title('Residual of estimation');
		ax.XLabel.String = 'time [s]';
		ax.YLabel.String = 'amplitude';
		
		
		a=figure;
		a.Name = 'Eigenvalues_of_system';
		subplot(1,2,1);
		viscircles([0 0], 1, 'LineStyle', ':' , 'LineWidth' , 0.5 , 'Color' , 'black');
		hold on;
		p = plot(real(eig_z),imag(eig_z));
		p.Marker = 'x';
 		p.LineStyle = 'none';
		p.MarkerSize = 9;
		p.Color = 'k';
		p.LineWidth = 2;
		hold off;
		ax=gca;
		ax.XLim = [-1.1, 1.1];
		ax.YLim = [-1.1, 1.1];
		ax.XAxisLocation  = 'origin';
		ax.YAxisLocation  = 'origin';
		ax.Box = 'on';
		title('Discrete eigenvalues');
		ax.XLabel.String = 'Re\{z\}';
		ax.YLabel.String = 'Im\{z\}';
		
		subplot(1,2,2);
		p = plot(real(eig_lambda),imag(eig_lambda));
		p.Marker = 'x';
		p.MarkerSize = 9;
		p.Color = 'k';
 		p.LineStyle = 'none';
		p.LineWidth = 2;
		ax=gca;
		ax.XLim = [-1.1*max(abs(real(eig_lambda))),10];
% 		ax.YLim = [-1.1, 1.1];
		ax.XAxisLocation  = 'origin';
		ax.YAxisLocation  = 'origin';
		ax.Box = 'on';
		title('Continous eigenvalues');
		ax.XLabel.String = ' Re\{\lambda\} ' ;
		ax.YLabel.String = ' Im\{\lambda\} ' ;
		ax.XGrid = 'on';
		ax.YGrid = 'on';
	
	
	

	
	%**************************************************************************************************											
	%% Output display formating:
	%**************************************************************************************************											


	line1='**************************************************************************************************\n';
	text = 'Complex exponential fit completed\n\nMSE = %.3f\n\n';

	fprintf([line1 text], MSE);

	text1= "Discrete_eig";
	text2 = "Continous_eig";
	text3 = "Prony_Coeff";

	matrix = [text2, text1, text3];
	matrix = convertStringsToChars(matrix);
	
	tab = table(eig_lambda,eig_z,cProny, 'VariableNames',matrix);
		
	disp(tab);

	
	end
	
			

		

	
	
	
	
	
	