
#
source("coef_OSAv1.0.R")
#
OSA_v1.0 <- function(PZENITH,PWIND,ZCHL){

#---------------------------------------------------------------------------------
# Initiliazing :
#---------------------------------------------------------------------------------	
INI=1
PDIR_ALB=numeric(INI)
PSCA_ALB=numeric(INI)
PWHC_ALB=numeric(INI)
PWHC_SFC=numeric(INI)
#
ZR22=numeric(INI)
ZSIG=numeric(INI)
ZUE=numeric(INI)
ZUE2=numeric(INI)
ZUE3=numeric(INI)
ZR11DF=numeric(INI)
ZFWC=numeric(INI)
ZNU=numeric(INI)
ZBP550=numeric(INI)         # computation variables
# 
ZWORK1=numeric(INI)
ZWORK2=numeric(INI)
ZWORK3=numeric(INI)
ZWORK4=numeric(INI)
ZWORK5=numeric(INI)                   # work array
#
ZCOSZEN=numeric(INI)
ZCOSZEN2=numeric(INI)
ZCOSZEN3=numeric(INI)      # Cosine of the zenith solar angle
#
ZAP=matrix(NA,INI,NNWL)
ZBBP=matrix(NA,INI,NNWL)
ZHB =matrix(NA,INI,NNWL)                  # computation variables
ZRR0=matrix(NA,INI,NNWL)
ZRRR=matrix(NA,INI,NNWL)
ZR00 =matrix(NA,INI,NNWL)          # computation variables
ZR11=matrix(NA,INI,NNWL)
ZRDF=matrix(NA,INI,NNWL)
ZRW=matrix(NA,INI,NNWL)
ZRWDF=matrix(NA,INI,NNWL)           # 4 components of the OSA
#
ZAKREFM2=numeric(NNWL)
ZYLMD =numeric(NNWL)                 # coeffs
# 
#
#
#
# * averaged global values for surface chlorophyll
#   (need to include bgc coupling in earth system model configuration)
#
#
#---------------------------------------------------------------------------------
# 0- Compute baseline values
#---------------------------------------------------------------------------------
#
# * compute the cosine of the solar zenith angle
#
ZCOSZEN = cos(PZENITH)
#
# * Compute sigma derived from wind speed (Cox & Munk reflectance model)
#
ZSIG = sqrt(0.003+0.00512*PWIND)
#
# * Correction for foam Monahanand and Muircheartaigh (1980) Eq 16-17
#   new: Salisbury 2014 eq(2) at 37GHz, value in fraction
#   has to be update once we have information from wave model (discussion with G. Madec)
#
ZFWC = 3.97E-4*exp(1.59*log(PWIND))
#
# * Backscattering by chlorophyll
#
ZYLMD = exp(0.014*(440.0-XAKWL))
#
# * uniform incidence of shortwave at surface (ue)
#
ZUE =rep(XUE,INI)
ZUE2=rep(XUE,INI)**2
ZUE3=rep(XUE,INI)**3
#
#---------------------------------------------------------------------------------
# 1- Compute direct surface albedo ZR11
#---------------------------------------------------------------------------------
#
ZAKREFM2 = XAKREFM*XAKREFM
ZCOSZEN2 = ZCOSZEN*ZCOSZEN
ZCOSZEN3 = ZCOSZEN*ZCOSZEN*ZCOSZEN
#
ZWORK4=0.0152-1.7873*ZCOSZEN+6.8972*ZCOSZEN2-8.5778*ZCOSZEN3+4.071*ZSIG-7.6446*ZCOSZEN*ZSIG
ZWORK5=exp(0.1643-7.8409*ZCOSZEN-3.5639*ZCOSZEN2-2.3588*ZSIG+10.0538*ZCOSZEN*ZSIG)
#
for ( JWL in 1:NNWL ){
for ( JI in 1:INI ){
#
      ZWORK1[JI]=sqrt(1.0-(1.0-ZCOSZEN2[JI])/ZAKREFM2[JWL])
#
    a = sqrt(1.0 - (1.0 - mu**2)/n_lambda**2)
    b = a-n_lambda*mu/(a+n_lambda*mu)
    b = b**2
    
    c = (mu-n_lambda*a)/(mu+n_lambda*c)
    c=c**2
    zrro=0.5*(b+c)
    
      ZWORK2[JI]=(ZWORK1[JI]-XAKREFM[JWL]*ZCOSZEN[JI])/(ZWORK1[JI]+XAKREFM[JWL]*ZCOSZEN[JI])
      ZWORK2[JI]=ZWORK2[JI]*ZWORK2[JI]
      ZWORK3[JI]=(ZCOSZEN[JI]-XAKREFM[JWL]*ZWORK1[JI])/(ZCOSZEN[JI]+XAKREFM[JWL]*ZWORK1[JI])
      ZWORK3[JI]=ZWORK3[JI]*ZWORK3[JI]
#
      ZRR0[JI,JWL]=0.50*(ZWORK2[JI]+ZWORK3[JI])
#  
      ZWORK2[JI]=(ZWORK1[JI]-1.34*ZCOSZEN[JI])/(ZWORK1[JI]+1.34*ZCOSZEN[JI])
      ZWORK2[JI]=ZWORK2[JI]*ZWORK2[JI]
      ZWORK3[JI]=(ZCOSZEN[JI]-1.34*ZWORK1[JI])/(ZCOSZEN[JI]+1.34*ZWORK1[JI])
      ZWORK3[JI]=ZWORK3[JI]*ZWORK3[JI]
#       
      ZRRR[JI,JWL]=0.50*(ZWORK2[JI]+ZWORK3[JI])
#  
#     direct albedo
      ZR11[JI,JWL]=ZRR0[JI,JWL]-ZWORK4[JI]*ZWORK5[JI]*ZRR0[JI,JWL]/ZRRR[JI,JWL]
#
  }
}
#
#---------------------------------------------------------------------------------
# 2- Compute surface diffuse albedo ZRDF
#---------------------------------------------------------------------------------
#
# * Diffuse albedo from Jin et al., 2006 (Eq 5b) 
#
for ( JWL in 1:NNWL ){
for ( JI in 1:INI ){
      ZRDF[JI,JWL] = -0.1479 + 0.1502*XAKREFM[JWL] - 0.0176*ZSIG[JI]*XAKREFM[JWL]
  }
}
#
#---------------------------------------------------------------------------------
# 3- Compute direct water-leaving albedo ZRW
#---------------------------------------------------------------------------------
#
# * Chlorophyll derived values
#
###########################################################
###########################################################
#when Chlorophyll will be coupled ZCHL SHOULD BE ZCHL
ZWORK4= exp(log(ZCHL)*0.65)
ZWORK5= log10(ZCHL)  
ZBP550= 0.416 * exp(log(ZCHL)*0.766) 
###########################################################
###########################################################
#
# * Direct reflectance partitioning based on Morel & Gentilli 1991
#
ZR22=0.48168549-0.014894708*ZSIG-0.20703885*ZSIG*ZSIG
#
for ( JWL in 1:NNWL ){
for ( JI in 1:INI ){
#
#     Determine absorption and backscattering
#     coefficients to determine reflectance below the surface (Ro) once for all
      ZAP[JI,JWL] = 0.06*XAKACHL[JWL]*ZWORK4[JI] + 0.2*(XAW440+0.06*ZWORK4[JI])*ZYLMD[JWL]
#
        ZNU   [JI]     = 0.5*(ZWORK5[JI]-0.3)
        ZWORK1[JI]     = exp(ZNU[JI]*log(XAKWL[JWL]/550.))
        ZBBP  [JI,JWL] = (0.002+0.01*(0.5-0.25*ZWORK5[JI])*ZWORK1[JI])*ZBP550[JI]
 #
#     Morel-Gentili(1991), Eq (12)
      ZHB[JI,JWL]=0.5*XAKBW[JWL]/(0.5*XAKBW[JWL]+ZBBP[JI,JWL])
#
#     Use Morel 91 formula to compute the direct reflectance below the surface
      ZWORK2[JI]=(0.5*XAKBW[JWL]+ZBBP[JI,JWL])/(XAKAW3[JWL]+ZAP[JI,JWL])
      ZWORK3[JI]=(0.6279-0.2227*ZHB[JI,JWL]-0.0513*ZHB[JI,JWL]*ZHB[JI,JWL] &
                -0.3119*ZCOSZEN[JI])+0.2465*ZHB[JI,JWL]*ZCOSZEN[JI]
      ZR00[JI,JWL]=ZWORK2[JI]*ZWORK3[JI]
#
#     water-leaving albedo
      ZRW[JI,JWL]=ZR00[JI,JWL]*(1.-ZR22[JI])/(1.-ZR00[JI,JWL]*ZR22[JI])
#
  }
}
#
#---------------------------------------------------------------------------------
# 4- Compute diffuse water-leaving albedo ZRWDF
#---------------------------------------------------------------------------------
#
ZWORK4=0.0152-1.7873*ZUE+6.8972*ZUE2-8.5778*ZUE3+4.071*ZSIG-7.6446*ZUE*ZSIG
ZWORK5=exp(0.1643-7.8409*ZUE-3.5639*ZUE2-2.3588*ZSIG+10.0538*ZUE*ZSIG)
#
for ( JWL in 1:NNWL ){
for ( JI in 1:INI ){
#
#     as previous water-leaving computation but assumes a uniform incidence of shortwave at surface (ue)
#
      ZWORK1[JI]=sqrt(1.0-(1.0-ZUE2[JI])/ZAKREFM2[JWL])    
#
      ZWORK2[JI]=(ZWORK1[JI]-XAKREFM[JWL]*ZUE[JI])/(ZWORK1[JI]+XAKREFM[JWL]*ZUE[JI])
      ZWORK2[JI]=ZWORK2[JI]*ZWORK2[JI]
      ZWORK3[JI]=(ZUE[JI]-XAKREFM[JWL]*ZWORK1[JI])/(ZUE[JI]+XAKREFM[JWL]*ZWORK1[JI])
      ZWORK3[JI]=ZWORK3[JI]*ZWORK3[JI]   
#
      ZRR0[JI,JWL]=0.50*(ZWORK2[JI]+ZWORK3[JI])
#
      ZWORK2[JI]=(ZWORK1[JI]-1.34*ZUE[JI])/(ZWORK1[JI]+1.34*ZUE[JI])
      ZWORK2[JI]=ZWORK2[JI]*ZWORK2[JI]
      ZWORK3[JI]=(ZUE[JI]-1.34*ZWORK1[JI])/(ZUE[JI]+1.34*ZWORK1[JI])
      ZWORK3[JI]=ZWORK3[JI]*ZWORK3[JI]   
#
      ZRRR[JI,JWL]=0.50*(ZWORK2[JI]+ZWORK3[JI])
      ZR11DF[JI]=ZRR0[JI,JWL]-ZWORK4[JI]*ZWORK5[JI]*ZRR0[JI,JWL]/ZRRR[JI,JWL]
#
#     Use Morel 91 formula to compute the diffuse
#     reflectance below the surface
      ZWORK2[JI]=(0.5*XAKBW[JWL]+ZBBP[JI,JWL])/(XAKAW3[JWL]+ZAP[JI,JWL])
      ZWORK3[JI]=0.6279-0.2227*ZHB[JI,JWL]-0.0513*ZHB[JI,JWL]*ZHB[JI,JWL] &
                -0.3119*ZUE[JI]+0.2465*ZHB[JI,JWL]*ZUE[JI]
      ZR00[JI,JWL]=ZWORK2[JI]*ZWORK3[JI]
#
#     diffuse albedo
      ZRWDF[JI,JWL]=ZR00[JI,JWL]*(1.-ZR22[JI])*(1.-ZR11DF[JI])/(1.-ZR00[JI,JWL]*ZR22[JI])
#
  }
} 
#
#---------------------------------------------------------------------------------
# 5- OSA estimation
#---------------------------------------------------------------------------------
#
# partitionning direct and diffuse albedo and accounts for foam spectral properties
#
for ( JWL in 1:NNWL ){
for ( JI in 1:INI ){
      #PDIR_ALB[JI] = PDIR_ALB[JI] + XFRWL[JWL] * ((1.-ZFWC[JI])*(ZR11[JI,JWL]+ZRW  [JI,JWL]))
      PDIR_ALB[JI] = PDIR_ALB[JI] + XFRWL[JWL] * ((ZR11[JI,JWL]+ZRW  [JI,JWL]))
      #PSCA_ALB[JI] = PSCA_ALB[JI] + XFRWL[JWL] * ((1.-ZFWC[JI])*(ZRDF[JI,JWL]+ZRWDF[JI,JWL]))
      PSCA_ALB[JI] = PSCA_ALB[JI] + XFRWL[JWL] * ((ZRDF[JI,JWL]+ZRWDF[JI,JWL]))
      #PWHC_ALB[JI] = PWHC_ALB[JI] + XFRWL[JWL] * (ZFWC[JI]*XRWC[JWL])
      PWHC_ALB[JI] = PWHC_ALB[JI] + XFRWL[JWL] * (XRWC[JWL])
      PWHC_SFC[JI] = ZFWC[JI]
  } 
} 
#

out=c(PDIR_ALB,PSCA_ALB,PWHC_ALB,PWHC_SFC)

return(out)


}

