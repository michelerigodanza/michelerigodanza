%% Caricamento Tabelle da excel
Madrid_NO2 = readtable("Madrid\Madrid_NO2.xlsx");
Madrid_PM10=readtable("Madrid\Madrid_PM10.xlsx");
Madrid_PM25 = readtable("Madrid\Madrid_PM2.5.xlsx");

London_NO2 = readtable("LONDON\NO2 LONDON\LONDON NO2.xlsx");
London_PM10 = readtable("LONDON\PM10 LONDON\LONDON PM10.xlsx");
London_PM25 = readtable("LONDON\PM25 LONDON\LONDON PM25.xlsx");

Berlin_NO2 = readtable("BERLIN\NO2 BERLIN\NO2 BERLIN.xlsx");
Berlin_PM10 = readtable("BERLIN\PM10 BERLIN\BERLIN PM10.xlsx");
Berlin_PM25 = readtable("BERLIN\PM25 BERLIN\BERLIN PM25.xlsx");

Milan_NO2=readtable("Milan\Milan_NO2.xlsx");
Milan_PM10=readtable("Milan\Milan_PM10.xlsx");
Milan_PM25 = readtable("Milan\Milan_PM2.5.xlsx");


%% Caricamento covid index

covid_index_Spain = readtable("INDICI COVID\covid-index-Spain.xlsx");
covid_index_Italy = readtable("INDICI COVID\covid-index-Italy.xlsx");
covid_index_UK = readtable("INDICI COVID\covid-index-United Kingdom.xlsx");
covid_index_Germany = readtable("INDICI COVID\covid-index-Germany.xlsx");

%% Caricamento dati meteo

meteo_berlin = readtable("BERLIN\Dati Meteo\Berlin 2020-01-01 to 2021-12-31.xlsx");
meteo_milan = readtable("Milan\Dati Meteo\Milan 2020-01-01 to 2021-12-31.xlsx");
meteo_london = readtable("LONDON\Dati Meteo\London 2020-01-01 to 2021-12-31.xlsx");
meteo_madrid = readtable("Madrid\Dati Meteo\madrid 2020-01-01 to 2021-12-31.xlsx");


%% Analisi Madrid
%Creazione timetable
MedieGior_NO2_Madrid = retime(timetable(Madrid_NO2.DatetimeBegin, Madrid_NO2.Concentration),'daily','mean');
MedieGior_PM10_Madrid = retime(timetable(Madrid_PM10.DatetimeBegin, Madrid_PM10.Concentration),'daily','mean');
MedieGior_PM25_Madrid = retime(timetable(Madrid_PM25.DatetimeBegin, Madrid_PM25.Concentration),'daily','mean');

%Eliminazione Trend NO2
a_MA_N02 = MedieGior_NO2_Madrid.Var1;
y_MA_N02 = detrend(a_MA_N02, 2, 'omitnan');
trend_MA_N02 = a_MA_N02 - y_MA_N02;

%Eliminazione Trend PM10
a_MA_PM10 = MedieGior_PM10_Madrid.Var1;
y_MA_PM10 = detrend(a_MA_PM10, 2, 'omitnan');
trend_MA_PM10 = a_MA_PM10 - y_MA_PM10;

%Eliminazione Trend PM25
a_MA_PM25 = MedieGior_PM25_Madrid.Var1;
y_MA_PM25 = detrend(a_MA_PM25, 2, 'omitnan');
trend_MA_PM25 = a_MA_PM25 - y_MA_PM25;

%Plot N02
figure(1)
subplot(2,3,1)
plot(MedieGior_NO2_Madrid.Time, MedieGior_NO2_Madrid.Var1);
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
title('Media giornaliera NO2 Madrid')

%Plot PM10
subplot(2,3,2)
plot(MedieGior_PM10_Madrid.Time, MedieGior_PM10_Madrid.Var1);
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
title('Media giornaliera PM10 Madrid')

%Plot PM25
subplot(2,3,3)
plot(MedieGior_PM25_Madrid.Time, MedieGior_PM25_Madrid.Var1);
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
title('Media giornaliera PM2.5 Madrid')


%Plot N02 detrend
subplot(2,3,4)
hold on
plot(MedieGior_NO2_Madrid.Time, trend_MA_N02, 'r');
plot(MedieGior_NO2_Madrid.Time, y_MA_N02, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off


%Plot PM10 detrend
subplot(2,3,5)
hold on
plot(MedieGior_PM10_Madrid.Time, trend_MA_PM10, 'r');
plot(MedieGior_PM10_Madrid.Time, y_MA_PM10, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%Plot PM25 detrend
subplot(2,3,6)
hold on
plot(MedieGior_PM25_Madrid.Time, trend_MA_PM25, 'r');
plot(MedieGior_PM25_Madrid.Time, y_MA_PM25, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off


%% Analisi Milano
%Creazione timetable
MedieGior_NO2_Milan = retime(timetable(Milan_NO2.DatetimeBegin, Milan_NO2.Concentration),'daily','mean');
MedieGior_PM10_Milan = retime(timetable(Milan_PM10.DatetimeBegin, Milan_PM10.Concentration),'daily','mean');
MedieGior_PM25_Milan = retime(timetable(Milan_PM25.DatetimeBegin, Milan_PM25.Concentration),'daily','mean');


%Eliminazione Trend NO2
a_MI_N02 = MedieGior_NO2_Milan.Var1;
y_MI_N02 = detrend(a_MI_N02, 2, 'omitnan');
trend_MI_N02 = a_MI_N02 - y_MI_N02;

%Eliminazione Trend PM10
a_MI_PM10 = MedieGior_PM10_Milan.Var1;
y_MI_PM10 = detrend(a_MI_PM10, 2, 'omitnan');
trend_MI_PM10 = a_MI_PM10 - y_MI_PM10;

%Eliminazione Trend PM25
a_MI_PM25 = MedieGior_PM25_Milan.Var1;
y_MI_PM25 = detrend(a_MI_PM25, 2, 'omitnan');
trend_MI_PM25 = a_MI_PM25 - y_MI_PM25;

%Plot N02
figure(2)
subplot(2,3,1)
plot(MedieGior_NO2_Milan.Time, MedieGior_NO2_Milan.Var1);
ylabel('');
xlabel('Giorni');
title('Media giornaliera NO2 Milan')

%Plot PM10
subplot(2,3,2)
plot(MedieGior_PM10_Milan.Time, MedieGior_PM10_Milan.Var1);
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
title('Media giornaliera PM10 Milan')

%Plot PM25
subplot(2,3,3)
plot(MedieGior_PM25_Milan.Time, MedieGior_PM25_Milan.Var1);
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
title('Media giornaliera PM2.5 Milan')


%Plot N02 detrend
subplot(2,3,4)
hold on
plot(MedieGior_NO2_Milan.Time, trend_MI_N02, 'r');
plot(MedieGior_NO2_Milan.Time, y_MI_N02, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off


%Plot PM10 detrend
subplot(2,3,5)
hold on
plot(MedieGior_PM10_Milan.Time, trend_MI_PM10, 'r');
plot(MedieGior_PM10_Milan.Time, y_MI_PM10, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%Plot PM25 detrend
subplot(2,3,6)
hold on
plot(MedieGior_PM25_Milan.Time, trend_MI_PM25, 'r');
plot(MedieGior_PM25_Milan.Time, y_MI_PM25, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off


%% Analisi Berlino
%Creazione timetable giornaliere
MedieGior_NO2_Berlin = retime(timetable(Berlin_NO2.DatetimeBegin, Berlin_NO2.Concentration),'daily','mean');
MedieGior_PM10_Berlin = retime(timetable(Berlin_PM10.DatetimeBegin, Berlin_PM10.Concentration),'daily','mean');
MedieGior_PM25_Berlin= retime(timetable(Berlin_PM25.DatetimeBegin, Berlin_PM25.Concentration),'daily','mean');

%Eliminazione trend NO2
a_BE_NO2 = MedieGior_NO2_Berlin.Var1;
y_BE_NO2 = detrend(a_BE_NO2, 2, 'omitnan');
trend_BE_NO2 = a_BE_NO2 - y_BE_NO2;

%Eliminazione trend PM10
a_BE_PM10 = MedieGior_PM10_Berlin.Var1;
y_BE_PM10 = detrend(a_BE_PM10, 2, 'omitnan');
trend_BE_PM10 = a_BE_PM10 - y_BE_PM10;

%Eliminazione trend PM 2.5
a_BE_PM25 = MedieGior_PM25_Berlin.Var1;
y_BE_PM25 = detrend(a_BE_PM25, 2, 'omitnan');
trend_BE_PM25 = a_BE_PM25 - y_BE_PM25;

%Plot NO2
figure(3)
subplot(2,3,1)
plot(MedieGior_NO2_Berlin.Time, MedieGior_NO2_Berlin.Var1);
title('Media giornaliera NO2 Berlino')

%Plot PM10
subplot(2,3,2)
plot(MedieGior_PM10_Berlin.Time, MedieGior_PM10_Berlin.Var1);
title('Media giornaliera PM10 Berlino')

%Plot PM25
subplot(2,3,3)
plot(MedieGior_PM25_Berlin.Time, MedieGior_PM25_Berlin.Var1);
title('Media giornaliera PM2.5 Berlino')

%Plot NO2 detrend
subplot(2,3,4)
hold on
plot(MedieGior_NO2_Berlin.Time, trend_BE_NO2, 'r');
plot(MedieGior_NO2_Berlin.Time, y_BE_NO2, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%Plot PM10 detrend
subplot(2,3,5)
hold on
plot(MedieGior_PM10_Berlin.Time, trend_BE_PM10, 'r');
plot(MedieGior_PM10_Berlin.Time, y_BE_PM10, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%Plot PM25 detrend
subplot(2,3,6)
hold on
plot(MedieGior_PM25_Berlin.Time, trend_BE_PM25, 'r');
plot(MedieGior_PM25_Berlin.Time, y_BE_PM25, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off


%% Analisi Londra
%Creazione timetable giornaliere
MedieGior_NO2_Londra = retime(timetable(London_NO2.DatetimeBegin, London_NO2.Concentration),'daily','mean');
MedieGior_PM10_Londra = retime(timetable(London_PM10.DatetimeBegin, London_PM10.Concentration),'daily','mean');
MedieGior_PM25_Londra= retime(timetable(London_PM25.DatetimeBegin, London_PM25.Concentration),'daily','mean');

%Eliminazione trend NO2
a_LO_NO2 = MedieGior_NO2_Londra.Var1;
y_LO_NO2 = detrend(a_LO_NO2, 2, 'omitnan');
trend_LO_NO2 = a_LO_NO2 - y_LO_NO2;

%Eliminazione trend PM10
a_LO_PM10 = MedieGior_PM10_Londra.Var1;
y_LO_PM10 = detrend(a_LO_PM10, 2, 'omitnan');
trend_LO_PM10 = a_LO_PM10 - y_LO_PM10;

%Eliminazione trend PM 2.5
a_LO_PM25 = MedieGior_PM25_Londra.Var1;
y_LO_PM25 = detrend(a_LO_PM25, 2, 'omitnan');
trend_LO_PM25 = a_LO_PM25 - y_LO_PM25;

%Plot NO2
figure(4)
subplot(2,3,1)
plot(MedieGior_NO2_Londra.Time, MedieGior_NO2_Londra.Var1);
title('Media giornaliera NO2 Londra')

%Plot PM10
subplot(2,3,2)
plot(MedieGior_PM10_Londra.Time, MedieGior_PM10_Londra.Var1);
title('Media giornaliera PM10 Londra')
%Plot PM25
subplot(2,3,3)
plot(MedieGior_PM25_Londra.Time, MedieGior_PM25_Londra.Var1);
title('Media giornaliera PM 2.5 Londra')

%Plot NO2 detrend
subplot(2,3,4)
hold on
plot(MedieGior_NO2_Londra.Time, trend_LO_NO2, 'r');
plot(MedieGior_NO2_Londra.Time, y_LO_NO2, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%Plot PM10 detrend
subplot(2,3,5)
hold on
plot(MedieGior_PM10_Londra.Time, trend_LO_PM10, 'r');
plot(MedieGior_PM10_Londra.Time, y_LO_PM10, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%Plot PM25 detrend
subplot(2,3,6)
hold on
plot(MedieGior_PM25_Londra.Time, trend_LO_PM25, 'r');
plot(MedieGior_PM25_Londra.Time, y_LO_PM25, 'b');
ylabel('Concentrazione: μg/m^{3}');
xlabel('Giorni');
legend('Fit detrend', 'Serie detrendizzata');
hold off

%% regressione lineare multipla con dati meteo
%berlin NO2
berlin_x1=covid_index_Germany.stringency_index(1:162);
berlin_x2=meteo_berlin.windspeed(21:182);
berlin_x3=meteo_berlin.temp(21:182);
berlin_x4=meteo_berlin.humidity(21:182);
berlin_x5=meteo_berlin.precip(21:182);
berlin_x6=meteo_berlin.visibility(21:182);
berlin_NO2_y=MedieGior_NO2_Berlin.Var1(1847:2008);


berlin_X = [ones(size(berlin_x2)) berlin_x2 berlin_x3 berlin_x2.*berlin_x3 berlin_x2.^2 berlin_x3.^2 berlin_x4 berlin_x5 berlin_x4.*berlin_x5 berlin_x5.^2 berlin_x6];
berlin_NO2_b = regress(berlin_NO2_y,berlin_X)    % Removes NaN data
[b_berlin_NO2,bint_berlin_NO2,residui_berlin_NO2,rint_berlin_NO2,stats_berlin_NO2] = regress(berlin_NO2_y,berlin_X)


figure (8)
s=scatter3(berlin_x2,berlin_x3, berlin_NO2_y)
s.Marker= 'none';

hold on
berlin_x1fit = min(berlin_x1):1:max(berlin_x1);
berlin_x2fit = min(berlin_x2):1:max(berlin_x2);
berlin_x3fit = min(berlin_x3):1:max(berlin_x3);
berlin_x4fit = min(berlin_x4):1:max(berlin_x4);
berlin_x5fit = min(berlin_x5):1:max(berlin_x5);
berlin_x6fit = min(berlin_x6):1:max(berlin_x6);
[berlin_X2FIT,berlin_X3FIT] = meshgrid(berlin_x2fit,berlin_x3fit);
[berlin_X4FIT,berlin_X5FIT] = meshgrid(berlin_x4fit,berlin_x5fit);
berlin_NO2_YFIT = berlin_NO2_b(1) + berlin_NO2_b(2)*berlin_X2FIT + berlin_NO2_b(3)*berlin_X3FIT + berlin_NO2_b(4)*berlin_X2FIT.*berlin_X3FIT + berlin_NO2_b(5)*berlin_X2FIT.^2 + berlin_NO2_b(6)*berlin_X3FIT.^2;
mesh(berlin_X2FIT,berlin_X3FIT,berlin_NO2_YFIT)
title('Regressione multipla NO2 Berlino', 'R^2='+ string(stats_berlin_NO2(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione NO_2: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(berlin_x4,berlin_x5, berlin_NO2_y,'filled')
s.Marker= 'none';
hold on
berlin_NO2_YFIT2 = berlin_NO2_b(1) + berlin_NO2_b(7)*berlin_X4FIT + berlin_NO2_b(8)*berlin_X5FIT + berlin_NO2_b(9)*berlin_X4FIT.*berlin_X5FIT + berlin_NO2_b(10)*berlin_X5FIT.^2;
mesh(berlin_X4FIT,berlin_X5FIT,berlin_NO2_YFIT2)
title('Regressione multipla NO2 Berlino', 'R^2='+ string(stats_berlin_NO2(1)))
xlabel('umidità: %')
ylabel('precipitazioni: mm')
zlabel('Concentrazione NO_2: μg/m^{3}')
view(50,10)
hold off


%% berlin PM10


berlin_PM10_y=MedieGior_PM10_Berlin.Var1(1847:2008);


berlin_X = [ones(size(berlin_x2)) berlin_x2 berlin_x3 berlin_x2.*berlin_x3 berlin_x2.^2 berlin_x3.^2 berlin_x4 berlin_x5 berlin_x4.*berlin_x5 berlin_x5.^2 berlin_x6];
berlin_PM10_b = regress(berlin_PM10_y,berlin_X)    % Removes NaN data
[b_berlin_PM10,bint_berlin_PM10,residui_berlin_PM10,rint_berlin_PM10,stats_berlin_PM10] = regress(berlin_PM10_y,berlin_X)


figure (8)
s=scatter3(berlin_x2,berlin_x3, berlin_PM10_y)
s.Marker= 'none';

hold on
berlin_x1fit = min(berlin_x1):1:max(berlin_x1);
berlin_x2fit = min(berlin_x2):1:max(berlin_x2);
berlin_x3fit = min(berlin_x3):1:max(berlin_x3);
berlin_x4fit = min(berlin_x4):1:max(berlin_x4);
berlin_x5fit = min(berlin_x5):1:max(berlin_x5);
berlin_x6fit = min(berlin_x6):1:max(berlin_x6);
[berlin_X2FIT,berlin_X3FIT] = meshgrid(berlin_x2fit,berlin_x3fit);
[berlin_X4FIT,berlin_X5FIT] = meshgrid(berlin_x4fit,berlin_x5fit);
berlin_PM10_YFIT = berlin_PM10_b(1) + berlin_PM10_b(2)*berlin_X2FIT + berlin_PM10_b(3)*berlin_X3FIT + berlin_PM10_b(4)*berlin_X2FIT.*berlin_X3FIT + berlin_PM10_b(5)*berlin_X2FIT.^2 + berlin_PM10_b(6)*berlin_X3FIT.^2;
mesh(berlin_X2FIT,berlin_X3FIT,berlin_PM10_YFIT)
title('Regressione multipla PM10 Berlino', 'R^2='+ string(stats_berlin_PM10(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM10: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(berlin_x4,berlin_x5, berlin_PM10_y,'filled')
s.Marker= 'none';
hold on
berlin_PM10_YFIT2 = berlin_PM10_b(1) + berlin_PM10_b(7)*berlin_X4FIT + berlin_PM10_b(8)*berlin_X5FIT + berlin_PM10_b(9)*berlin_X4FIT.*berlin_X5FIT + berlin_PM10_b(10)*berlin_X5FIT.^2;
mesh(berlin_X4FIT,berlin_X5FIT,berlin_PM10_YFIT2)
title('Regressione multipla PM10 Berlino', 'R^2='+ string(stats_berlin_PM10(1)))
xlabel('umidità: %')
ylabel('precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off



%% berlin PM2.5


berlin_PM25_y=MedieGior_PM25_Berlin.Var1(1847:2008);


berlin_X = [ones(size(berlin_x2)) berlin_x2 berlin_x3 berlin_x2.*berlin_x3 berlin_x2.^2 berlin_x3.^2 berlin_x4 berlin_x5 berlin_x4.*berlin_x5 berlin_x5.^2 berlin_x6];
berlin_PM25_b = regress(berlin_PM25_y,berlin_X)    % Removes NaN data
[b_berlin_PM25,bint_berlin_PM25,residui_berlin_PM25,rint_berlin_PM25,stats_berlin_PM25] = regress(berlin_PM25_y,berlin_X)


figure (8)
s=scatter3(berlin_x2,berlin_x3, berlin_PM25_y)
s.Marker= 'none';

hold on
berlin_x1fit = min(berlin_x1):1:max(berlin_x1);
berlin_x2fit = min(berlin_x2):1:max(berlin_x2);
berlin_x3fit = min(berlin_x3):1:max(berlin_x3);
berlin_x4fit = min(berlin_x4):1:max(berlin_x4);
berlin_x5fit = min(berlin_x5):1:max(berlin_x5);
berlin_x6fit = min(berlin_x6):1:max(berlin_x6);
[berlin_X2FIT,berlin_X3FIT] = meshgrid(berlin_x2fit,berlin_x3fit);
[berlin_X4FIT,berlin_X5FIT] = meshgrid(berlin_x4fit,berlin_x5fit);
berlin_PM25_YFIT = berlin_PM25_b(1) + berlin_PM25_b(2)*berlin_X2FIT + berlin_PM25_b(3)*berlin_X3FIT + berlin_PM25_b(4)*berlin_X2FIT.*berlin_X3FIT + berlin_PM25_b(5)*berlin_X2FIT.^2 + berlin_PM25_b(6)*berlin_X3FIT.^2;
mesh(berlin_X2FIT,berlin_X3FIT,berlin_PM25_YFIT)
title('Regressione multipla PM25 Berlino', 'R^2='+ string(stats_berlin_PM25(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM25: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(berlin_x4,berlin_x5, berlin_PM25_y,'filled')
s.Marker= 'none';
hold on
berlin_PM25_YFIT2 = berlin_PM25_b(1) + berlin_PM25_b(7)*berlin_X4FIT + berlin_PM25_b(8)*berlin_X5FIT + berlin_PM25_b(9)*berlin_X4FIT.*berlin_X5FIT + berlin_PM25_b(10)*berlin_X5FIT.^2;
mesh(berlin_X4FIT,berlin_X5FIT,berlin_PM25_YFIT2)
title('Regressione multipla PM25 Berlino', 'R^2='+ string(stats_berlin_PM25(1)))
xlabel('umidità: %')
ylabel('precipitazioni: mm')
zlabel('Concentrazione PM25: μg/m^{3}')
view(50,10)
hold off

%% madrid NO2
madrid_x1=covid_index_Spain.stringency_index(1:162);
madrid_x2=meteo_madrid.windspeed(21:182);
madrid_x3=meteo_madrid.temp(21:182);
madrid_x4=meteo_madrid.humidity(21:182);
madrid_x5=meteo_madrid.precip(21:182);
madrid_x6=meteo_madrid.visibility(21:182);
madrid_NO2_y=MedieGior_NO2_Madrid.Var1(1847:2008);


madrid_X = [ones(size(madrid_x2)) madrid_x2 madrid_x3 madrid_x2.*madrid_x3 madrid_x2.^2 madrid_x3.^2 madrid_x4 madrid_x5 madrid_x4.*madrid_x5 madrid_x5.^2 madrid_x6];
madrid_NO2_b = regress(madrid_NO2_y,madrid_X)    % Removes NaN data
[b_madrid_NO2,bint_madrid_NO2,residui_madrid_NO2,rint_madrid_NO2,stats_madrid_NO2] = regress(madrid_NO2_y,madrid_X)


figure (8)
s=scatter3(madrid_x2,madrid_x3, madrid_NO2_y)
s.Marker= 'none';

hold on
madrid_x1fit = min(madrid_x1):1:max(madrid_x1);
madrid_x2fit = min(madrid_x2):1:max(madrid_x2);
madrid_x3fit = min(madrid_x3):1:max(madrid_x3);
madrid_x4fit = min(madrid_x4):1:max(madrid_x4);
madrid_x5fit = min(madrid_x5):1:max(madrid_x5);
madrid_x6fit = min(madrid_x6):1:max(madrid_x6);
[madrid_X2FIT,madrid_X3FIT] = meshgrid(madrid_x2fit,madrid_x3fit);
[madrid_X4FIT,madrid_X5FIT] = meshgrid(madrid_x4fit,madrid_x5fit);
madrid_NO2_YFIT = madrid_NO2_b(1) + madrid_NO2_b(2)*madrid_X2FIT + madrid_NO2_b(3)*madrid_X3FIT + madrid_NO2_b(4)*madrid_X2FIT.*madrid_X3FIT + madrid_NO2_b(5)*madrid_X2FIT.^2 + madrid_NO2_b(6)*madrid_X3FIT.^2;
mesh(madrid_X2FIT,madrid_X3FIT,madrid_NO2_YFIT)
title('Regressione multipla NO_2 Madrid', 'R^2='+ string(stats_madrid_NO2(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione NO2: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(madrid_x4,madrid_x5, madrid_NO2_y,'filled')
s.Marker= 'none';
hold on
madrid_NO2_YFIT2 = madrid_NO2_b(1) + madrid_NO2_b(7)*madrid_X4FIT + madrid_NO2_b(8)*madrid_X5FIT + madrid_NO2_b(9)*madrid_X4FIT.*madrid_X5FIT + madrid_NO2_b(10)*madrid_X5FIT.^2;
mesh(madrid_X4FIT,madrid_X5FIT,madrid_NO2_YFIT2)
title('Regressione multipla NO_2 Madrid', 'R^2='+ string(stats_madrid_NO2(1)))
xlabel('umidità: %')
ylabel('precipitazioni: mm')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off


%% madrid PM10
madrid_PM10_y=MedieGior_PM10_Madrid.Var1(1847:2008);


madrid_X = [ones(size(madrid_x2)) madrid_x2 madrid_x3 madrid_x2.*madrid_x3 madrid_x2.^2 madrid_x3.^2 madrid_x4 madrid_x5 madrid_x4.*madrid_x5 madrid_x5.^2 madrid_x6];
madrid_PM10_b = regress(madrid_PM10_y,madrid_X)    % Removes NaN data
[b_madrid_PM10,bint_madrid_PM10,residui_madrid_PM10,rint_madrid_PM10,stats_madrid_PM10] = regress(madrid_PM10_y,madrid_X)


figure (8)
s=scatter3(madrid_x2,madrid_x3, madrid_PM10_y)
s.Marker= 'none';

hold on
madrid_x1fit = min(madrid_x1):1:max(madrid_x1);
madrid_x2fit = min(madrid_x2):1:max(madrid_x2);
madrid_x3fit = min(madrid_x3):1:max(madrid_x3);
madrid_x4fit = min(madrid_x4):1:max(madrid_x4);
madrid_x5fit = min(madrid_x5):1:max(madrid_x5);
madrid_x6fit = min(madrid_x6):1:max(madrid_x6);
[madrid_X2FIT,madrid_X3FIT] = meshgrid(madrid_x2fit,madrid_x3fit);
[madrid_X4FIT,madrid_X5FIT] = meshgrid(madrid_x4fit,madrid_x5fit);
madrid_PM10_YFIT = madrid_PM10_b(1) + madrid_PM10_b(2)*madrid_X2FIT + madrid_PM10_b(3)*madrid_X3FIT + madrid_PM10_b(4)*madrid_X2FIT.*madrid_X3FIT + madrid_PM10_b(5)*madrid_X2FIT.^2 + madrid_PM10_b(6)*madrid_X3FIT.^2;
mesh(madrid_X2FIT,madrid_X3FIT,madrid_PM10_YFIT)
title('Regressione multipla PM10 Madrid', 'R^2='+ string(stats_madrid_PM10(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM10: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(madrid_x4,madrid_x5, madrid_PM10_y,'filled')
s.Marker= 'none';
hold on
madrid_PM10_YFIT2 = madrid_PM10_b(1) + madrid_PM10_b(7)*madrid_X4FIT + madrid_PM10_b(8)*madrid_X5FIT + madrid_PM10_b(9)*madrid_X4FIT.*madrid_X5FIT + madrid_PM10_b(10)*madrid_X5FIT.^2;
mesh(madrid_X4FIT,madrid_X5FIT,madrid_PM10_YFIT2)
title('Regressione multipla PM10 Madrid', 'R^2='+ string(stats_madrid_PM10(1)))
xlabel('umidità: %')
ylabel('precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

%% madrid PM2.5

madrid_PM25_y=MedieGior_PM25_Madrid.Var1(1847:2008);


madrid_X = [ones(size(madrid_x2)) madrid_x2 madrid_x3 madrid_x2.*madrid_x3 madrid_x2.^2 madrid_x3.^2 madrid_x4 madrid_x5 madrid_x4.*madrid_x5 madrid_x5.^2 madrid_x6];
madrid_PM25_b = regress(madrid_PM25_y,madrid_X)    % Removes NaN data
[b_madrid_PM25,bint_madrid_PM25,residui_madrid_PM25,rint_madrid_PM25,stats_madrid_PM25] = regress(madrid_PM25_y,madrid_X)


figure (8)
s=scatter3(madrid_x2,madrid_x3, madrid_PM25_y)
s.Marker= 'none';

hold on
madrid_x1fit = min(madrid_x1):1:max(madrid_x1);
madrid_x2fit = min(madrid_x2):1:max(madrid_x2);
madrid_x3fit = min(madrid_x3):1:max(madrid_x3);
madrid_x4fit = min(madrid_x4):1:max(madrid_x4);
madrid_x5fit = min(madrid_x5):1:max(madrid_x5);
madrid_x6fit = min(madrid_x6):1:max(madrid_x6);
[madrid_X2FIT,madrid_X3FIT] = meshgrid(madrid_x2fit,madrid_x3fit);
[madrid_X4FIT,madrid_X5FIT] = meshgrid(madrid_x4fit,madrid_x5fit);
madrid_PM25_YFIT = madrid_PM25_b(1) + madrid_PM25_b(2)*madrid_X2FIT + madrid_PM25_b(3)*madrid_X3FIT + madrid_PM25_b(4)*madrid_X2FIT.*madrid_X3FIT + madrid_PM25_b(5)*madrid_X2FIT.^2 + madrid_PM25_b(6)*madrid_X3FIT.^2;
mesh(madrid_X2FIT,madrid_X3FIT,madrid_PM25_YFIT)
title('Regressione multipla PM2.5 Madrid', 'R^2='+ string(stats_madrid_PM25(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM25: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(madrid_x4,madrid_x5, madrid_PM25_y,'filled')
s.Marker= 'none';
hold on
madrid_PM25_YFIT2 = madrid_PM25_b(1) + madrid_PM25_b(7)*madrid_X4FIT + madrid_PM25_b(8)*madrid_X5FIT + madrid_PM25_b(9)*madrid_X4FIT.*madrid_X5FIT + madrid_PM25_b(10)*madrid_X5FIT.^2;
mesh(madrid_X4FIT,madrid_X5FIT,madrid_PM25_YFIT2)
title('Regressione multipla PM2.5 Madrid', 'R^2='+ string(stats_madrid_PM25(1)))
xlabel('umidità: %')
ylabel('precipitazioni: mm')
zlabel('Concentrazione PM25: μg/m^{3}')
view(50,10)
hold off

%% london NO2
london_x1=covid_index_Germany.stringency_index(1:162);
london_x2=meteo_london.windspeed(21:182);
london_x3=meteo_london.temp(21:182);
london_x4=meteo_london.humidity(21:182);
london_x5=meteo_london.precip(21:182);
london_x6=meteo_london.visibility(21:182);
london_NO2_y=MedieGior_NO2_Londra.Var1(1847:2008);


london_X = [ones(size(london_x2)) london_x2 london_x3 london_x2.*london_x3 london_x2.^2 london_x3.^2 london_x4 london_x5 london_x4.*london_x5 london_x5.^2 london_x6];
london_NO2_b = regress(london_NO2_y,london_X)    % Removes NaN data
[b_london_NO2,bint_london_NO2,residui_london_NO2,rint_london_NO2,stats_london_NO2] = regress(london_NO2_y,london_X)


figure (8)
s=scatter3(london_x2,london_x3, london_NO2_y)
s.Marker= 'none';

hold on
london_x1fit = min(london_x1):1:max(london_x1);
london_x2fit = min(london_x2):1:max(london_x2);
london_x3fit = min(london_x3):1:max(london_x3);
london_x4fit = min(london_x4):1:max(london_x4);
london_x5fit = min(london_x5):1:max(london_x5);
london_x6fit = min(london_x6):1:max(london_x6);
[london_X2FIT,london_X3FIT] = meshgrid(london_x2fit,london_x3fit);
[london_X4FIT,london_X5FIT] = meshgrid(london_x4fit,london_x5fit);
london_NO2_YFIT = london_NO2_b(1) + london_NO2_b(2)*london_X2FIT + london_NO2_b(3)*london_X3FIT + london_NO2_b(4)*london_X2FIT.*london_X3FIT + london_NO2_b(5)*london_X2FIT.^2 + london_NO2_b(6)*london_X3FIT.^2;
mesh(london_X2FIT,london_X3FIT,london_NO2_YFIT)
title('Regressione multipla NO_2 London', 'R^2='+ string(stats_london_NO2(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione NO_2: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(london_x4,london_x5, london_NO2_y,'filled')
s.Marker= 'none';
hold on
london_NO2_YFIT2 = london_NO2_b(1) + london_NO2_b(7)*london_X4FIT + london_NO2_b(8)*london_X5FIT + london_NO2_b(9)*london_X4FIT.*london_X5FIT + london_NO2_b(10)*london_X5FIT.^2;
mesh(london_X4FIT,london_X5FIT,london_NO2_YFIT2)
title('Regressione multipla NO_2 London', 'R^2='+ string(stats_london_NO2(1)))
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione NO_2: μg/m^{3}')
view(500,10)
hold off

%% london PM10
london_PM10_y=MedieGior_PM10_Londra.Var1(1847:2008);
london_X = [ones(size(london_x2)) london_x2 london_x3 london_x2.*london_x3 london_x2.^2 london_x3.^2 london_x4 london_x5 london_x4.*london_x5 london_x5.^2 london_x6];

london_PM10_b = regress(london_PM10_y,london_X)    % Removes NaN data
[b_london_PM10,bint_london_PM10,residui_london_PM10,rint_london_PM10,stats_london_PM10] = regress(london_PM10_y,london_X)


figure (8)
s=scatter3(london_x2,london_x3, london_PM10_y)
s.Marker= 'none';

hold on
london_x1fit = min(london_x1):1:max(london_x1);
london_x2fit = min(london_x2):1:max(london_x2);
london_x3fit = min(london_x3):1:max(london_x3);
london_x4fit = min(london_x4):1:max(london_x4);
london_x5fit = min(london_x5):1:max(london_x5);
london_x6fit = min(london_x6):1:max(london_x6);
[london_X2FIT,london_X3FIT] = meshgrid(london_x2fit,london_x3fit);
[london_X4FIT,london_X5FIT] = meshgrid(london_x4fit,london_x5fit);
london_PM10_YFIT = london_PM10_b(1) + london_PM10_b(2)*london_X2FIT + london_PM10_b(3)*london_X3FIT + london_PM10_b(4)*london_X2FIT.*london_X3FIT + london_PM10_b(5)*london_X2FIT.^2 + london_PM10_b(6)*london_X3FIT.^2;
mesh(london_X2FIT,london_X3FIT,london_PM10_YFIT)
title('Regressione multipla PM10 Londra', 'R^2='+ string(stats_london_PM10(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM10: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(london_x4,london_x5, london_PM10_y,'filled')
s.Marker= 'none';
hold on
london_PM10_YFIT2 = london_PM10_b(1) + london_PM10_b(7)*london_X4FIT + london_PM10_b(8)*london_X5FIT + london_PM10_b(9)*london_X4FIT.*london_X5FIT + london_PM10_b(10)*london_X5FIT.^2;
mesh(london_X4FIT,london_X5FIT,london_PM10_YFIT2)
title('Regressione multipla PM10 Londra', 'R^2='+ string(stats_london_PM10(1)))
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(500,10)
hold off

%% london PM25
london_PM25_y=MedieGior_PM25_Londra.Var1(1847:2008);
london_X = [ones(size(london_x2)) london_x2 london_x3 london_x2.*london_x3 london_x2.^2 london_x3.^2 london_x4 london_x5 london_x4.*london_x5 london_x5.^2 london_x6];

london_PM25_b = regress(london_PM25_y,london_X)    % Removes NaN data
[b_london_PM25,bint_london_PM25,residui_london_PM25,rint_london_PM25,stats_london_PM25] = regress(london_PM25_y,london_X)


figure (8)
s=scatter3(london_x2,london_x3, london_PM25_y)
s.Marker= 'none';

hold on
london_x1fit = min(london_x1):1:max(london_x1);
london_x2fit = min(london_x2):1:max(london_x2);
london_x3fit = min(london_x3):1:max(london_x3);
london_x4fit = min(london_x4):1:max(london_x4);
london_x5fit = min(london_x5):1:max(london_x5);
london_x6fit = min(london_x6):1:max(london_x6);
[london_X2FIT,london_X3FIT] = meshgrid(london_x2fit,london_x3fit);
[london_X4FIT,london_X5FIT] = meshgrid(london_x4fit,london_x5fit);
london_PM25_YFIT = london_PM25_b(1) + london_PM25_b(2)*london_X2FIT + london_PM25_b(3)*london_X3FIT + london_PM25_b(4)*london_X2FIT.*london_X3FIT + london_PM25_b(5)*london_X2FIT.^2 + london_PM25_b(6)*london_X3FIT.^2;
mesh(london_X2FIT,london_X3FIT,london_PM25_YFIT)
title('Regressione multipla PM2.5 Londra', 'R^2='+ string(stats_london_PM25(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM2.5: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(london_x4,london_x5, london_PM25_y,'filled')
s.Marker= 'none';
hold on
london_PM25_YFIT2 = london_PM25_b(1) + london_PM25_b(7)*london_X4FIT + london_PM25_b(8)*london_X5FIT + london_PM25_b(9)*london_X4FIT.*london_X5FIT + london_PM25_b(10)*london_X5FIT.^2;
mesh(london_X4FIT,london_X5FIT,london_PM25_YFIT2)
title('Regressione multipla PM2.5 Londra', 'R^2='+ string(stats_london_PM25(1)))
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(500,10)
hold off


%% milan NO2
milan_x1=covid_index_Germany.stringency_index(1:162);
milan_x2=meteo_milan.windspeed(21:182);
milan_x3=meteo_milan.temp(21:182);
milan_x4=meteo_milan.humidity(21:182);
milan_x5=meteo_milan.precip(21:182);
milan_x6=meteo_milan.visibility(21:182);
milan_NO2_y=MedieGior_NO2_Milan.Var1(1847:2008);


milan_X = [ones(size(milan_x2)) milan_x2 milan_x3 milan_x2.*milan_x3 milan_x2.^2 milan_x3.^2 milan_x4 milan_x5 milan_x4.*milan_x5 milan_x5.^2 milan_x6];
milan_NO2_b = regress(milan_NO2_y,milan_X)    % Removes NaN data
[b_milan_NO2,bint_milan_NO2,residui_milan_NO2,rint_milan_NO2,stats_milan_NO2] = regress(milan_NO2_y,milan_X)


figure (8)
s=scatter3(milan_x2,milan_x3, milan_NO2_y)
s.Marker= 'none';

hold on
milan_x1fit = min(milan_x1):1:max(milan_x1);
milan_x2fit = min(milan_x2):1:max(milan_x2);
milan_x3fit = min(milan_x3):1:max(milan_x3);
milan_x4fit = min(milan_x4):1:max(milan_x4);
milan_x5fit = min(milan_x5):1:max(milan_x5);
milan_x6fit = min(milan_x6):1:max(milan_x6);
[milan_X2FIT,milan_X3FIT] = meshgrid(milan_x2fit,milan_x3fit);
[milan_X4FIT,milan_X5FIT] = meshgrid(milan_x4fit,milan_x5fit);
milan_NO2_YFIT = milan_NO2_b(1) + milan_NO2_b(2)*milan_X2FIT + milan_NO2_b(3)*milan_X3FIT + milan_NO2_b(4)*milan_X2FIT.*milan_X3FIT + milan_NO2_b(5)*milan_X2FIT.^2 + milan_NO2_b(6)*milan_X3FIT.^2;
mesh(milan_X2FIT,milan_X3FIT,milan_NO2_YFIT)
title('Regressione multipla NO_2 Milano', 'R^2='+ string(stats_milan_NO2(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione NO_2: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(milan_x4,milan_x5, milan_NO2_y,'filled')
s.Marker= 'none';
hold on
milan_NO2_YFIT2 = milan_NO2_b(1) + milan_NO2_b(7)*milan_X4FIT + milan_NO2_b(8)*milan_X5FIT + milan_NO2_b(9)*milan_X4FIT.*milan_X5FIT + milan_NO2_b(10)*milan_X5FIT.^2;
mesh(milan_X4FIT,milan_X5FIT,milan_NO2_YFIT2)
title('Regressione multipla NO_2 Milano', 'R^2='+ string(stats_milan_NO2(1)))
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione NO_2: μg/m^{3}')
view(500,10)
hold off


%% milan PM10
milan_PM10_y=MedieGior_PM10_Milan.Var1(1847:2008);
milan_X = [ones(size(milan_x2)) milan_x2 milan_x3 milan_x2.*milan_x3 milan_x2.^2 milan_x3.^2 milan_x4 milan_x5 milan_x4.*milan_x5 milan_x5.^2 milan_x6];

milan_PM10_b = regress(milan_PM10_y,milan_X)    % Removes NaN data
[b_milan_PM10,bint_milan_PM10,residui_milan_PM10,rint_milan_PM10,stats_milan_PM10] = regress(milan_PM10_y,milan_X)


figure (8)
s=scatter3(milan_x2,milan_x3, milan_PM10_y)
s.Marker= 'none';

hold on
milan_x1fit = min(milan_x1):1:max(milan_x1);
milan_x2fit = min(milan_x2):1:max(milan_x2);
milan_x3fit = min(milan_x3):1:max(milan_x3);
milan_x4fit = min(milan_x4):1:max(milan_x4);
milan_x5fit = min(milan_x5):1:max(milan_x5);
milan_x6fit = min(milan_x6):1:max(milan_x6);
[milan_X2FIT,milan_X3FIT] = meshgrid(milan_x2fit,milan_x3fit);
[milan_X4FIT,milan_X5FIT] = meshgrid(milan_x4fit,milan_x5fit);
milan_PM10_YFIT = milan_PM10_b(1) + milan_PM10_b(2)*milan_X2FIT + milan_PM10_b(3)*milan_X3FIT + milan_PM10_b(4)*milan_X2FIT.*milan_X3FIT + milan_PM10_b(5)*milan_X2FIT.^2 + milan_PM10_b(6)*milan_X3FIT.^2;
mesh(milan_X2FIT,milan_X3FIT,milan_PM10_YFIT)
title('Regressione multipla PM10 Milano', 'R^2='+ string(stats_milan_PM10(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM10: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(milan_x4,milan_x5, milan_PM10_y,'filled')
s.Marker= 'none';
hold on
milan_PM10_YFIT2 = milan_PM10_b(1) + milan_PM10_b(7)*milan_X4FIT + milan_PM10_b(8)*milan_X5FIT + milan_PM10_b(9)*milan_X4FIT.*milan_X5FIT + milan_PM10_b(10)*milan_X5FIT.^2;
mesh(milan_X4FIT,milan_X5FIT,milan_PM10_YFIT2)
title('Regressione multipla PM10 Milano', 'R^2='+ string(stats_milan_PM10(1)))
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(500,10)
hold off






%% milan PM25
milan_PM25_y=MedieGior_PM25_Milan.Var1(1847:2008);
milan_X = [ones(size(milan_x2)) milan_x2 milan_x3 milan_x2.*milan_x3 milan_x2.^2 milan_x3.^2 milan_x4 milan_x5 milan_x4.*milan_x5 milan_x5.^2 milan_x6];

milan_PM25_b = regress(milan_PM25_y,milan_X)    % Removes NaN data
[b_milan_PM25,bint_milan_PM25,residui_milan_PM25,rint_milan_PM25,stats_milan_PM25] = regress(milan_PM25_y,milan_X)


figure (8)
s=scatter3(milan_x2,milan_x3, milan_PM25_y)
s.Marker= 'none';

hold on
milan_x1fit = min(milan_x1):1:max(milan_x1);
milan_x2fit = min(milan_x2):1:max(milan_x2);
milan_x3fit = min(milan_x3):1:max(milan_x3);
milan_x4fit = min(milan_x4):1:max(milan_x4);
milan_x5fit = min(milan_x5):1:max(milan_x5);
milan_x6fit = min(milan_x6):1:max(milan_x6);
[milan_X2FIT,milan_X3FIT] = meshgrid(milan_x2fit,milan_x3fit);
[milan_X4FIT,milan_X5FIT] = meshgrid(milan_x4fit,milan_x5fit);
milan_PM25_YFIT = milan_PM25_b(1) + milan_PM25_b(2)*milan_X2FIT + milan_PM25_b(3)*milan_X3FIT + milan_PM25_b(4)*milan_X2FIT.*milan_X3FIT + milan_PM25_b(5)*milan_X2FIT.^2 + milan_PM25_b(6)*milan_X3FIT.^2;
mesh(milan_X2FIT,milan_X3FIT,milan_PM25_YFIT)
title('Regressione multipla PM2.5 Milano', 'R^2='+ string(stats_milan_PM25(1)))
xlabel('Vento: km/h')
ylabel('Temperatura: C°')
zlabel('Concentrazione PM2.5: μg/ m^3')
view(50,10)
hold off

figure (9)
s=scatter3(milan_x4,milan_x5, milan_PM25_y,'filled')
s.Marker= 'none';
hold on
milan_PM25_YFIT2 = milan_PM25_b(1) + milan_PM25_b(7)*milan_X4FIT + milan_PM25_b(8)*milan_X5FIT + milan_PM25_b(9)*milan_X4FIT.*milan_X5FIT + milan_PM25_b(10)*milan_X5FIT.^2;
mesh(milan_X4FIT,milan_X5FIT,milan_PM25_YFIT2)
title('Regressione multipla PM2.5 Milano', 'R^2='+ string(stats_milan_PM25(1)))
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(500,10)
hold off







%% Regressione eliminato l'effetto meteo
%Berlino
reg_berlin_NO2 = fitlm(berlin_x1 ,residui_berlin_NO2);
reg_berlin_PM10 = fitlm(berlin_x1 ,residui_berlin_PM10);
reg_berlin_PM25 = fitlm(berlin_x1 ,residui_berlin_PM25);

%Madrid
reg_madrid_NO2 = fitlm(madrid_x1 ,residui_madrid_NO2);
reg_madrid_PM10 = fitlm(madrid_x1 ,residui_madrid_PM10);
reg_madrid_PM25 = fitlm(madrid_x1 ,residui_madrid_PM25);

%Londra
reg_london_NO2 = fitlm(london_x1 ,residui_london_NO2);
reg_london_PM10 = fitlm(london_x1 ,residui_london_PM10);
reg_london_PM25 = fitlm(london_x1 ,residui_london_PM25);

%Milano
reg_milan_NO2 = fitlm(milan_x1 ,residui_milan_NO2);
reg_milan_PM10 = fitlm(milan_x1 ,residui_milan_PM10);
reg_milan_PM25 = fitlm(milan_x1 ,residui_milan_PM25);

%% PLOT
%Berlin NO2
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(berlin_x2,berlin_x3, berlin_NO2_y,'filled')
hold on
berlin_x1fit = min(berlin_x1):1:max(berlin_x1);
berlin_x2fit = min(berlin_x2):1:max(berlin_x2);
berlin_x3fit = min(berlin_x3):1:max(berlin_x3);
berlin_x4fit = min(berlin_x4):1:max(berlin_x4);
berlin_x5fit = min(berlin_x5):1:max(berlin_x5);
berlin_x6fit = min(berlin_x6):1:max(berlin_x6);
[berlin_X2FIT,berlin_X3FIT] = meshgrid(berlin_x2fit,berlin_x3fit);
[berlin_X4FIT,berlin_X5FIT] = meshgrid(berlin_x4fit,berlin_x5fit);
berlin_NO2_YFIT = berlin_NO2_b(1) + berlin_NO2_b(2)*berlin_X2FIT + berlin_NO2_b(3)*berlin_X3FIT;
mesh(berlin_X2FIT,berlin_X3FIT,berlin_NO2_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(berlin_x4,berlin_x5, berlin_NO2_y,'filled')
hold on
berlin_NO2_YFIT2 = berlin_NO2_b(1) + berlin_NO2_b(4)*berlin_X4FIT + berlin_NO2_b(5)*berlin_X5FIT;
mesh(berlin_X4FIT,berlin_X5FIT,berlin_NO2_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,3);
hold on
yyaxis left
h_berlin_NO2 = plot(fitresult_berlin_NO2, xData_berlin_NO2, yData_berlin_NO2 );
legend( h_berlin_NO2, 'residui berlin NO2 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_berlin_NO2', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,berlin_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_berlin_NO2);

%% Berlin PM10
subplot(2,2,1)
scatter3(berlin_x2,berlin_x3, berlin_PM10_y,'filled')
hold on
berlin_PM10_YFIT = berlin_PM10_b(1) + berlin_PM10_b(2)*berlin_X2FIT + berlin_PM10_b(3)*berlin_X3FIT;
mesh(berlin_X2FIT,berlin_X3FIT,berlin_PM10_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(berlin_x4,berlin_x5, berlin_PM10_y,'filled')
hold on
berlin_PM10_YFIT2 = berlin_PM10_b(1) + berlin_PM10_b(4)*berlin_X4FIT + berlin_PM10_b(5)*berlin_X5FIT;
mesh(berlin_X4FIT,berlin_X5FIT,berlin_PM10_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,3);
hold on
yyaxis left
h_berlin_PM10 = plot(fitresult_berlin_PM10, xData_berlin_PM10, yData_berlin_PM10 );
legend( h_berlin_PM10, 'residui berlin PM10 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_berlin_PM10', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,berlin_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_berlin_PM10);

%% Berlin PM2.5
subplot(2,2,1)
scatter3(berlin_x2,berlin_x3, berlin_PM25_y,'filled')
hold on
berlin_PM25_YFIT = berlin_PM25_b(1) + berlin_PM25_b(2)*berlin_X2FIT + berlin_PM25_b(3)*berlin_X3FIT;
mesh(berlin_X2FIT,berlin_X3FIT,berlin_PM25_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(berlin_x4,berlin_x5, berlin_PM25_y,'filled')
hold on
berlin_PM25_YFIT2 = berlin_PM25_b(1) + berlin_PM25_b(4)*berlin_X4FIT + berlin_PM25_b(5)*berlin_X5FIT;
mesh(berlin_X4FIT,berlin_X5FIT,berlin_PM25_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,3);
hold on
yyaxis left
h_berlin_PM25 = plot(fitresult_berlin_PM25, xData_berlin_PM25, yData_berlin_PM25 );
legend( h_berlin_PM25, 'residui berlin PM2.5 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_berlin_PM25', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,berlin_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_berlin_PM25);


%% PLOT
%London NO2
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(london_x2,london_x3, london_NO2_y,'filled')
hold on
london_x1fit = min(london_x1):1:max(london_x1);
london_x2fit = min(london_x2):1:max(london_x2);
london_x3fit = min(london_x3):1:max(london_x3);
london_x4fit = min(london_x4):1:max(london_x4);
london_x5fit = min(london_x5):1:max(london_x5);
london_x6fit = min(london_x6):1:max(london_x6);
[london_X2FIT,london_X3FIT] = meshgrid(london_x2fit,london_x3fit);
[london_X4FIT,london_X5FIT] = meshgrid(london_x4fit,london_x5fit);
london_NO2_YFIT = london_NO2_b(1) + london_NO2_b(2)*london_X2FIT + london_NO2_b(3)*london_X3FIT;
mesh(london_X2FIT,london_X3FIT,london_NO2_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(london_x4,london_x5, london_NO2_y,'filled')
hold on
london_NO2_YFIT2 = london_NO2_b(1) + london_NO2_b(4)*london_X4FIT + london_NO2_b(5)*london_X5FIT;
mesh(london_X4FIT,london_X5FIT,london_NO2_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,3);
hold on
yyaxis left
h_london_NO2 = plot(fitresult_london_NO2, xData_london_NO2, yData_london_NO2 );
legend( h_london_NO2, 'residui london NO2 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_london_NO2', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,london_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_london_NO2);


%% PLOT
%London PM10
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(london_x2,london_x3, london_PM10_y,'filled')
hold on
london_x1fit = min(london_x1):1:max(london_x1);
london_x2fit = min(london_x2):1:max(london_x2);
london_x3fit = min(london_x3):1:max(london_x3);
london_x4fit = min(london_x4):1:max(london_x4);
london_x5fit = min(london_x5):1:max(london_x5);
london_x6fit = min(london_x6):1:max(london_x6);
[london_X2FIT,london_X3FIT] = meshgrid(london_x2fit,london_x3fit);
[london_X4FIT,london_X5FIT] = meshgrid(london_x4fit,london_x5fit);
london_PM10_YFIT = london_PM10_b(1) + london_PM10_b(2)*london_X2FIT + london_PM10_b(3)*london_X3FIT;
mesh(london_X2FIT,london_X3FIT,london_PM10_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(london_x4,london_x5, london_PM10_y,'filled')
hold on
london_PM10_YFIT2 = london_PM10_b(1) + london_PM10_b(4)*london_X4FIT + london_PM10_b(5)*london_X5FIT;
mesh(london_X4FIT,london_X5FIT,london_PM10_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,3);
hold on
yyaxis left
h_london_PM10 = plot(fitresult_london_PM10, xData_london_PM10, yData_london_PM10 );
legend( h_london_PM10, 'residui london PM10 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_london_PM10', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,london_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_london_PM10);

%% PLOT
%London PM2.5
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(london_x2,london_x3, london_PM25_y,'filled')
hold on
london_x1fit = min(london_x1):1:max(london_x1);
london_x2fit = min(london_x2):1:max(london_x2);
london_x3fit = min(london_x3):1:max(london_x3);
london_x4fit = min(london_x4):1:max(london_x4);
london_x5fit = min(london_x5):1:max(london_x5);
london_x6fit = min(london_x6):1:max(london_x6);
[london_X2FIT,london_X3FIT] = meshgrid(london_x2fit,london_x3fit);
[london_X4FIT,london_X5FIT] = meshgrid(london_x4fit,london_x5fit);
london_PM25_YFIT = london_PM25_b(1) + london_PM25_b(2)*london_X2FIT + london_PM25_b(3)*london_X3FIT;
mesh(london_X2FIT,london_X3FIT,london_PM25_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(london_x4,london_x5, london_PM25_y,'filled')
hold on
london_PM25_YFIT2 = london_PM25_b(1) + london_PM25_b(4)*london_X4FIT + london_PM25_b(5)*london_X5FIT;
mesh(london_X4FIT,london_X5FIT,london_PM25_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,3);
hold on
yyaxis left
h_london_PM25 = plot(fitresult_london_PM25, xData_london_PM25, yData_london_PM25 );
legend( h_london_PM25, 'residui london PM2.5 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_london_PM2.5', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,london_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_london_PM25);


%% PLOT
%Madrid NO2
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(madrid_x2,madrid_x3, madrid_NO2_y,'filled')
hold on
madrid_x1fit = min(madrid_x1):1:max(madrid_x1);
madrid_x2fit = min(madrid_x2):1:max(madrid_x2);
madrid_x3fit = min(madrid_x3):1:max(madrid_x3);
madrid_x4fit = min(madrid_x4):1:max(madrid_x4);
madrid_x5fit = min(madrid_x5):1:max(madrid_x5);
madrid_x6fit = min(madrid_x6):1:max(madrid_x6);
[madrid_X2FIT,madrid_X3FIT] = meshgrid(madrid_x2fit,madrid_x3fit);
[madrid_X4FIT,madrid_X5FIT] = meshgrid(madrid_x4fit,madrid_x5fit);
madrid_NO2_YFIT = madrid_NO2_b(1) + madrid_NO2_b(2)*madrid_X2FIT + madrid_NO2_b(3)*madrid_X3FIT;
mesh(madrid_X2FIT,madrid_X3FIT,madrid_NO2_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura °C')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(madrid_x4,madrid_x5, madrid_NO2_y,'filled')
hold on
madrid_NO2_YFIT2 = madrid_NO2_b(1) + madrid_NO2_b(4)*madrid_X4FIT + madrid_NO2_b(5)*madrid_X5FIT;
mesh(madrid_X4FIT,madrid_X5FIT,madrid_NO2_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off


subplot(2,2,3);
hold on
yyaxis left
h_madrid_NO2 = plot(fitresult_madrid_NO2, xData_madrid_NO2, yData_madrid_NO2);
legend( h_madrid_NO2, 'residui madrid NO2 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_madrid_NO2', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,madrid_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_madrid_NO2);

%% Madrid PM10
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(madrid_x2,madrid_x3, madrid_PM10_y,'filled')
hold on
madrid_x1fit = min(madrid_x1):1:max(madrid_x1);
madrid_x2fit = min(madrid_x2):1:max(madrid_x2);
madrid_x3fit = min(madrid_x3):1:max(madrid_x3);
madrid_x4fit = min(madrid_x4):1:max(madrid_x4);
madrid_x5fit = min(madrid_x5):1:max(madrid_x5);
madrid_x6fit = min(madrid_x6):1:max(madrid_x6);
[madrid_X2FIT,madrid_X3FIT] = meshgrid(madrid_x2fit,madrid_x3fit);
[madrid_X4FIT,madrid_X5FIT] = meshgrid(madrid_x4fit,madrid_x5fit);
madrid_PM10_YFIT = madrid_PM10_b(1) + madrid_PM10_b(2)*madrid_X2FIT + madrid_PM10_b(3)*madrid_X3FIT;
mesh(madrid_X2FIT,madrid_X3FIT,madrid_PM10_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura °C')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(madrid_x4,madrid_x5, madrid_PM10_y,'filled')
hold on
madrid_PM10_YFIT2 = madrid_PM10_b(1) + madrid_PM10_b(4)*madrid_X4FIT + madrid_PM10_b(5)*madrid_X5FIT;
mesh(madrid_X4FIT,madrid_X5FIT,madrid_PM10_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off


subplot(2,2,3);
hold on
yyaxis left
h_madrid_PM10 = plot(fitresult_madrid_PM10, xData_madrid_PM10, yData_madrid_PM10 );
legend( h_madrid_PM10, 'residui madrid PM10 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_madrid_PM10', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,madrid_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_madrid_PM10);

%% Madrid PM25
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(madrid_x2,madrid_x3, madrid_PM25_y,'filled')
hold on
madrid_x1fit = min(madrid_x1):1:max(madrid_x1);
madrid_x2fit = min(madrid_x2):1:max(madrid_x2);
madrid_x3fit = min(madrid_x3):1:max(madrid_x3);
madrid_x4fit = min(madrid_x4):1:max(madrid_x4);
madrid_x5fit = min(madrid_x5):1:max(madrid_x5);
madrid_x6fit = min(madrid_x6):1:max(madrid_x6);
[madrid_X2FIT,madrid_X3FIT] = meshgrid(madrid_x2fit,madrid_x3fit);
[madrid_X4FIT,madrid_X5FIT] = meshgrid(madrid_x4fit,madrid_x5fit);
madrid_PM25_YFIT = madrid_PM25_b(1) + madrid_PM25_b(2)*madrid_X2FIT + madrid_PM25_b(3)*madrid_X3FIT;
mesh(madrid_X2FIT,madrid_X3FIT,madrid_PM25_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura °C')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(madrid_x4,madrid_x5, madrid_PM25_y,'filled')
hold on
madrid_PM25_YFIT2 = madrid_PM25_b(1) + madrid_PM25_b(4)*madrid_X4FIT + madrid_PM25_b(5)*madrid_X5FIT;
mesh(madrid_X4FIT,madrid_X5FIT,madrid_PM25_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off


subplot(2,2,3);
hold on
yyaxis left
h_madrid_PM25 = plot(fitresult_madrid_PM25, xData_madrid_PM25, yData_madrid_PM25 );
legend( h_madrid_PM25, 'residui Madrid PM2.5 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_madrid_PM25', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,madrid_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_madrid_PM25);



%% PLOT
%Milan NO2
XDates = [datetime(2020,1,21) :datetime(2020,6,30)];

subplot(2,2,1)
scatter3(milan_x2,milan_x3, milan_NO2_y,'filled')
hold on
milan_x1fit = min(milan_x1):1:max(milan_x1);
milan_x2fit = min(milan_x2):1:max(milan_x2);
milan_x3fit = min(milan_x3):1:max(milan_x3);
milan_x4fit = min(milan_x4):1:max(milan_x4);
milan_x5fit = min(milan_x5):1:max(milan_x5);
milan_x6fit = min(milan_x6):1:max(milan_x6);
[milan_X2FIT,milan_X3FIT] = meshgrid(milan_x2fit,milan_x3fit);
[milan_X4FIT,milan_X5FIT] = meshgrid(milan_x4fit,milan_x5fit);
milan_NO2_YFIT = milan_NO2_b(1) + milan_NO2_b(2)*milan_X2FIT + milan_NO2_b(3)*milan_X3FIT;
mesh(milan_X2FIT,milan_X3FIT,milan_NO2_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(milan_x4,milan_x5, milan_NO2_y,'filled')
hold on
milan_NO2_YFIT2 = milan_NO2_b(1) + milan_NO2_b(4)*milan_X4FIT + milan_NO2_b(5)*milan_X5FIT;
mesh(milan_X4FIT,milan_X5FIT,milan_NO2_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione NO2: μg/m^{3}')
view(50,10)
hold off


subplot(2,2,3);
hold on
yyaxis left
h_milan_NO2 = plot(fitresult_milan_NO2, xData_milan_NO2, yData_milan_NO2 );
legend( h_milan_NO2, 'residui milan NO2 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_milan_NO2', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,milan_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_milan_NO2);


%% Milan PM10
subplot(2,2,1)
scatter3(milan_x2,milan_x3, milan_PM10_y,'filled')
hold on
milan_x1fit = min(milan_x1):1:max(milan_x1);
milan_x2fit = min(milan_x2):1:max(milan_x2);
milan_x3fit = min(milan_x3):1:max(milan_x3);
milan_x4fit = min(milan_x4):1:max(milan_x4);
milan_x5fit = min(milan_x5):1:max(milan_x5);
milan_x6fit = min(milan_x6):1:max(milan_x6);
[milan_X2FIT,milan_X3FIT] = meshgrid(milan_x2fit,milan_x3fit);
[milan_X4FIT,milan_X5FIT] = meshgrid(milan_x4fit,milan_x5fit);
milan_PM10_YFIT = milan_PM10_b(1) + milan_PM10_b(2)*milan_X2FIT + milan_PM10_b(3)*milan_X3FIT;
mesh(milan_X2FIT,milan_X3FIT,milan_PM10_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(milan_x4,milan_x5, milan_PM10_y,'filled')
hold on
milan_PM10_YFIT2 = milan_PM10_b(1) + milan_PM10_b(4)*milan_X4FIT + milan_PM10_b(5)*milan_X5FIT;
mesh(milan_X4FIT,milan_X5FIT,milan_PM10_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM10: μg/m^{3}')
view(50,10)
hold off


subplot(2,2,3);
hold on
yyaxis left
h_milan_PM10 = plot(fitresult_milan_PM10, xData_milan_PM10, yData_milan_PM10 );
legend( h_milan_PM10, 'residui milan PM10 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_milan_PM10', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,milan_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_milan_PM10);

%% Milan PM25
subplot(2,2,1)
scatter3(milan_x2,milan_x3, milan_PM25_y,'filled')
hold on
milan_x1fit = min(milan_x1):1:max(milan_x1);
milan_x2fit = min(milan_x2):1:max(milan_x2);
milan_x3fit = min(milan_x3):1:max(milan_x3);
milan_x4fit = min(milan_x4):1:max(milan_x4);
milan_x5fit = min(milan_x5):1:max(milan_x5);
milan_x6fit = min(milan_x6):1:max(milan_x6);
[milan_X2FIT,milan_X3FIT] = meshgrid(milan_x2fit,milan_x3fit);
[milan_X4FIT,milan_X5FIT] = meshgrid(milan_x4fit,milan_x5fit);
milan_PM25_YFIT = milan_PM25_b(1) + milan_PM25_b(2)*milan_X2FIT + milan_PM25_b(3)*milan_X3FIT;
mesh(milan_X2FIT,milan_X3FIT,milan_PM25_YFIT)
title('Concentrazione spiegata da vento e temperatura')
xlabel('Vento: km/h')
ylabel('Temperatura: °C')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off

subplot(2,2,2);
scatter3(milan_x4,milan_x5, milan_PM25_y,'filled')
hold on
milan_PM25_YFIT2 = milan_PM25_b(1) + milan_PM25_b(4)*milan_X4FIT + milan_PM25_b(5)*milan_X5FIT;
mesh(milan_X4FIT,milan_X5FIT,milan_PM25_YFIT2)
title('Concentrazione spiegata da umidità e precipitazioni')
xlabel('Umidità: %')
ylabel('Precipitazioni: mm')
zlabel('Concentrazione PM2.5: μg/m^{3}')
view(50,10)
hold off


subplot(2,2,3);
hold on
yyaxis left
h_milan_PM25 = plot(fitresult_milan_PM25, xData_milan_PM25, yData_milan_PM25 );
legend( h_milan_PM25, 'residui milan PM25 vs. scaledDates', 'smooth dei residui');
% Label axes
xlabel( 'scaledDates', 'Interpreter', 'none' );
ylabel( 'residui_milan_PM2.5', 'Interpreter', 'none' );
datetick('x')
yyaxis right
s=plot(DateNumber,milan_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')

subplot(2,2,4);
plot(reg_milan_PM25);

%% pre smoothing date
dates = XDates
DateNumber = datenum(dates)

%% %% smoothing berlin NO2
[xData_berlin_NO2, yData_berlin_NO2] = prepareCurveData( DateNumber, residui_berlin_NO2 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_berlin_NO2, gof_berlin_NO2] = fit( xData_berlin_NO2, yData_berlin_NO2, ft, opts );

% Plot fit with data.
figure(32);
hold on
yyaxis left
h_berlin_NO2 = plot(fitresult_berlin_NO2, xData_berlin_NO2, yData_berlin_NO2 );
legend( h_berlin_NO2, 'Residui vs.Data giornaliera', 'Smooth di Resuidi NO2: μg/m^{3}');

% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi NO2: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,berlin_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing berlin PM10
[xData_berlin_PM10, yData_berlin_PM10] = prepareCurveData( DateNumber, residui_berlin_PM10 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_berlin_PM10, gof_berlin_PM10] = fit( xData_berlin_PM10, yData_berlin_PM10, ft, opts );

% Plot fit with data.
figure(33);

hold on
yyaxis left
h_berlin_PM10 = plot(fitresult_berlin_PM10, xData_berlin_PM10, yData_berlin_PM10 );
legend( h_berlin_PM10, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM10: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Residui PM10: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,berlin_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off


%% smoothing berlin PM25
[xData_berlin_PM25, yData_berlin_PM25] = prepareCurveData( DateNumber, residui_berlin_PM25 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_berlin_PM25, gof_berlin_PM25] = fit( xData_berlin_PM25, yData_berlin_PM25, ft, opts );

% Plot fit with data.
figure(34);
hold on
yyaxis left
h_berlin_PM25 = plot(fitresult_berlin_PM25, xData_berlin_PM25, yData_berlin_PM25 );
legend( h_berlin_PM25, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM2.5: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM2.5: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,berlin_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing madrid NO2

[xData_madrid_NO2, yData_madrid_NO2] = prepareCurveData( DateNumber, residui_madrid_NO2 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_madrid_NO2, gof_madrid_NO2] = fit( xData_madrid_NO2, yData_madrid_NO2, ft, opts );

% Plot fit with data.
figure(35);
hold on
yyaxis left
h_madrid_NO2 = plot(fitresult_madrid_NO2, xData_madrid_NO2, yData_madrid_NO2 );
legend( h_madrid_NO2, 'Residui vs.Data giornaliera', 'Smooth di Resuidi NO2 μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi NO2: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,madrid_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing madrid PM10
[xData_madrid_PM10, yData_madrid_PM10] = prepareCurveData( DateNumber, residui_madrid_PM10 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_madrid_PM10, gof_madrid_PM10] = fit( xData_madrid_PM10, yData_madrid_PM10, ft, opts );

% Plot fit with data.
figure(36);
hold on
yyaxis left
h_madrid_PM10 = plot(fitresult_madrid_PM10, xData_madrid_PM10, yData_madrid_PM10 );
legend( h_madrid_PM10, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM10: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM10: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,madrid_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing madrid PM25
[xData_madrid_PM25, yData_madrid_PM25] = prepareCurveData( DateNumber, residui_madrid_PM25 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_madrid_PM25, gof_madrid_PM25] = fit( xData_madrid_PM25, yData_madrid_PM25, ft, opts );

% Plot fit with data.
figure(37);
hold on
yyaxis left
h_madrid_PM25 = plot(fitresult_madrid_PM25, xData_madrid_PM25, yData_madrid_PM25 );
legend( h_madrid_PM25, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM2.5: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM2.5: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,madrid_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing london NO2

[xData_london_NO2, yData_london_NO2] = prepareCurveData( DateNumber, residui_london_NO2 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_london_NO2, gof_london_NO2] = fit( xData_london_NO2, yData_london_NO2, ft, opts );

% Plot fit with data.
figure(38);
hold on
yyaxis left
h_london_NO2 = plot(fitresult_london_NO2, xData_london_NO2, yData_london_NO2 );
legend( h_london_NO2, 'Residui vs.Data giornaliera', 'Smooth di Resuidi NO2: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi NO2: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,london_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing london PM10
[xData_london_PM10, yData_london_PM10] = prepareCurveData( DateNumber, residui_london_PM10 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_london_PM10, gof_london_PM10] = fit( xData_london_PM10, yData_london_PM10, ft, opts );

% Plot fit with data.
figure(39);
hold on
yyaxis left
h_london_PM10 = plot(fitresult_london_PM10, xData_london_PM10, yData_london_PM10 );
legend( h_london_PM10, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM10: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM10: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,london_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing london PM25
[xData_london_PM25, yData_london_PM25] = prepareCurveData( DateNumber, residui_london_PM25 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_london_PM25, gof_london_PM25] = fit( xData_london_PM25, yData_london_PM25, ft, opts );

% Plot fit with data.
figure(40);
hold on
yyaxis left
h_london_PM25 = plot(fitresult_london_PM25, xData_london_PM25, yData_london_PM25 );
legend( h_london_PM25, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM2.5: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM2.5: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,london_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing milan NO2

[xData_milan_NO2, yData_milan_NO2] = prepareCurveData( DateNumber, residui_milan_NO2 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_milan_NO2, gof_milan_NO2] = fit( xData_milan_NO2, yData_milan_NO2, ft, opts );

% Plot fit with data.
figure(41);
hold on
yyaxis left
h_milan_NO2 = plot(fitresult_milan_NO2, xData_milan_NO2, yData_milan_NO2 );
legend( h_milan_NO2, 'Residui vs.Data giornaliera', 'Smooth di Resuidi NO2: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi NO2: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,milan_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing milan PM10
[xData_milan_PM10, yData_milan_PM10] = prepareCurveData( DateNumber, residui_milan_PM10 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_milan_PM10, gof_milan_PM10] = fit( xData_milan_PM10, yData_milan_PM10, ft, opts );

% Plot fit with data.
figure(42);
hold on
yyaxis left
h_milan_PM10 = plot(fitresult_milan_PM10, xData_milan_PM10, yData_milan_PM10 );
legend( h_milan_PM10, 'Residui vs.Data giornaliera', 'Smooth di Resuidi concentrazione PM10: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM10: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,milan_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off

%% smoothing milan PM25
[xData_milan_PM25, yData_milan_PM25] = prepareCurveData( DateNumber, residui_milan_PM25 );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( 'Method', 'SmoothingSpline' );
opts.Normalize = 'on';
opts.SmoothingParam = 0.999998923596841;

% Fit model to data.
[fitresult_milan_PM25, gof_milan_PM25] = fit( xData_milan_PM25, yData_milan_PM25, ft, opts );

% Plot fit with data.
figure(43);
hold on
yyaxis left
h_milan_PM25 = plot(fitresult_milan_PM25, xData_milan_PM25, yData_milan_PM25 );
legend( h_milan_PM25, 'Residui vs.Data giornaliera', 'Smooth di Resuidi PM2.5: μg/m^{3}');
% Label axes
xlabel( ' ', 'Interpreter', 'none' );
ylabel( 'Resuidi PM2.5: μg/m^{3}');
datetick('x')
yyaxis right
s=plot(DateNumber,milan_x1, 'k');
ylabel( 'Indice rigidità lockdown: %', 'Interpreter', 'none' );
datetick('x')


grid on
hold off
