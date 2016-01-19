B = 1; F = 2; G = 3; D = 4; FT = 5;

names = cell(1,5);
names{B}  = 'Battery';
names{F}  = 'Fuel';
names{G}  = 'Gauge';
names{D}  = 'Distance';
names{FT} = 'BatteryFillTank';

dgm = zeros(5,5);
dgm(B,G) = 1;
dgm(F,G) = 1;
dgm(G,[D, FT]) = 1;

CPDs{B} = tabularCpdCreate([0.1 ; 0.9]);
CPDs{F} = tabularCpdCreate([0.1 ; 0.9]);
CPDs{G} = tabularCpdCreate(reshape([0.9 0.8 0.8 0.2 0.1 0.2 0.2 0.8], 2, 2, 2));