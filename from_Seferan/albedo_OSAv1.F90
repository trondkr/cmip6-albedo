!SFX_LIC Copyright 1994-2014 CNRS, Meteo-France and Universite Paul Sabatier
!SFX_LIC This is part of the SURFEX software governed by the CeCILL-C licence
!SFX_LIC version 1. See LICENSE, CeCILL-C_V1-en.txt and CeCILL-C_V1-fr.txt  
!SFX_LIC for details. version 1.
!     #########
 SUBROUTINE ALBEDO_OSAv1(PZENITH,PWIND,PDIR_ALB,PSCA_ALB)
!     ##################################################################
!
!!****  *ALBEDO_OSAv1*  
!!
!!    PURPOSE
!!    -------
!       computes the direct & diffuse albedo over open water
!
!     
!!**  METHOD
!!    ------
!
!!    EXTERNAL
!!    --------
!!
!!    IMPLICIT ARGUMENTS
!!    ------------------ 
!!
!!      
!!    REFERENCE
!!    ---------
!!
!!      
!!    AUTHOR
!!    ------
!!	R. Séférian           * Meteo-France *
!!
!!    MODIFICATIONS
!!    -------------
!!      Original    03/2014
!                   05/2014 R. Séférian & B. Decharme :: Adaptation to spectral
!                   computation for diffuse and direct albedo
!                   09/2014 R. Séférian & B. Sunghye :: Adaptation to spectral
!                   bands compatible with 6-bands RRTM radiative code
!                   09/2016 R. Séférian : optimizing algorithm
!                   01/2017 B. Decharme : optimizing algorithm
!       
!-------------------------------------------------------------------------------
!
!*           DECLARATIONS
!            ------------
!
USE MODD_ALBEDO_OSAv1_PAR
!
USE MODD_CSTS, ONLY : XPI
!
USE YOMHOOK   ,ONLY : LHOOK,   DR_HOOK
USE PARKIND1  ,ONLY : JPRB
!
IMPLICIT NONE
!
!*      0.1    declarations of arguments
!              -------------------------
!
REAL, DIMENSION(:), INTENT(IN)  :: PZENITH              ! zenithal angle (radian)
REAL, DIMENSION(:), INTENT(IN)  :: PWIND                ! surface wind (m s-1)
REAL, DIMENSION(:), INTENT(OUT) :: PDIR_ALB             ! direct  ocean surface albedo
REAL, DIMENSION(:), INTENT(OUT) :: PSCA_ALB             ! diffuse ocean surface albedo
!
!*      0.2    declarations of local variables
!              -------------------------
!
REAL, DIMENSION(SIZE(PZENITH))       :: ZCHL                             ! surface chlorophyll
!
REAL, DIMENSION(SIZE(PZENITH))       :: ZR22, ZSIG, ZUE, ZUE2, ZUE3,  &
                                        ZR11DF,ZFWC, ZNU, ZBP550         ! computation variables
! 
REAL, DIMENSION(SIZE(PZENITH))       :: ZWORK1, ZWORK2, ZWORK3, &
                                        ZWORK4, ZWORK5                   ! work array
!
REAL, DIMENSION(SIZE(PZENITH))       :: ZCOSZEN, ZCOSZEN2, ZCOSZEN3      ! Cosine of the zenith solar angle
!
REAL, DIMENSION(SIZE(PZENITH),NNWL)  :: ZAP, ZBBP, ZHB                   ! computation variables
REAL, DIMENSION(SIZE(PZENITH),NNWL)  :: ZRR0, ZRRR, ZR00           ! computation variables
REAL, DIMENSION(SIZE(PZENITH),NNWL)  :: ZR11, ZRDF, ZRW, ZRWDF           ! 4 components of the OSA
!
REAL, DIMENSION(NNWL)                :: ZAKREFM2, ZYLMD                  ! coeffs
! 
INTEGER                              :: INI, JI, JWL                     ! indexes
!
REAL(KIND=JPRB) :: ZHOOK_HANDLE
!
!-------------------------------------------------------------------------------
!
IF (LHOOK) CALL DR_HOOK('ALBEDO_OSAv1',0,ZHOOK_HANDLE)
!
!---------------------------------------------------------------------------------
! Initiliazing :
!---------------------------------------------------------------------------------
!
INI=SIZE(PZENITH)
!
PDIR_ALB(:) = 0. 
PSCA_ALB(:) = 0. 
!
! * averaged global values for surface chlorophyll
!   (need to include bgc coupling in earth system model configuration)
!
ZCHL(:) = 0.05
!
!---------------------------------------------------------------------------------
! 0- Compute baseline values
!---------------------------------------------------------------------------------
!
! * compute the cosine of the solar zenith angle
!
ZCOSZEN(:) = MAX(COS(PZENITH(:)),0.)
!
! * Compute sigma derived from wind speed (Cox & Munk reflectance model)
!
ZSIG(:) = SQRT(0.003+0.00512*PWIND(:))
!
! * Correction for foam Monahanand and Muircheartaigh (1980) Eq 16-17
!   new: Salisbury 2014 eq(2) at 37GHz, value in fraction
!   has to be update once we have information from wave model (discussion with G. Madec)
!
ZFWC(:) = 3.97E-4*EXP(1.59*LOG(PWIND(:)))
!
! * Backscattering by chlorophyll
!
ZYLMD(:) = EXP(0.014*(440.0-XAKWL(:)))
!
! * uniform incidence of shortwave at surface (ue)
!
ZUE (:)=XUE
ZUE2(:)=XUE**2
ZUE3(:)=XUE**3
!
!---------------------------------------------------------------------------------
! 1- Compute direct surface albedo ZR11
!---------------------------------------------------------------------------------
!
ZAKREFM2(:) = XAKREFM(:)*XAKREFM(:)
ZCOSZEN2(:) = ZCOSZEN(:)*ZCOSZEN(:)
ZCOSZEN3(:) = ZCOSZEN(:)*ZCOSZEN(:)*ZCOSZEN(:)
!
ZWORK4(:)=0.0152-1.7873*ZCOSZEN(:)+6.8972*ZCOSZEN2(:)-8.5778*ZCOSZEN3(:)+4.071*ZSIG(:)-7.6446*ZCOSZEN(:)*ZSIG(:)
ZWORK5(:)=EXP(0.1643-7.8409*ZCOSZEN(:)-3.5639*ZCOSZEN2(:)-2.3588*ZSIG(:)+10.0538*ZCOSZEN(:)*ZSIG(:))
!
DO JWL=1,NNWL
   DO JI=1,INI
!
      ZWORK1(JI)=SQRT(1.0-(1.0-ZCOSZEN2(JI))/ZAKREFM2(JWL))
!  
      ZWORK2(JI)=(ZWORK1(JI)-XAKREFM(JWL)*ZCOSZEN(JI))/(ZWORK1(JI)+XAKREFM(JWL)*ZCOSZEN(JI))
      ZWORK2(JI)=ZWORK2(JI)*ZWORK2(JI)
      ZWORK3(JI)=(ZCOSZEN(JI)-XAKREFM(JWL)*ZWORK1(JI))/(ZCOSZEN(JI)+XAKREFM(JWL)*ZWORK1(JI))
      ZWORK3(JI)=ZWORK3(JI)*ZWORK3(JI)
!
      ZRR0(JI,JWL)=0.50*(ZWORK2(JI)+ZWORK3(JI))
!  
      ZWORK2(JI)=(ZWORK1(JI)-1.34*ZCOSZEN(JI))/(ZWORK1(JI)+1.34*ZCOSZEN(JI))
      ZWORK2(JI)=ZWORK2(JI)*ZWORK2(JI)
      ZWORK3(JI)=(ZCOSZEN(JI)-1.34*ZWORK1(JI))/(ZCOSZEN(JI)+1.34*ZWORK1(JI))
      ZWORK3(JI)=ZWORK3(JI)*ZWORK3(JI)
!       
      ZRRR(JI,JWL)=0.50*(ZWORK2(JI)+ZWORK3(JI))
!  
!     direct albedo
      ZR11(JI,JWL)=ZRR0(JI,JWL)-ZWORK4(JI)*ZWORK5(JI)*ZRR0(JI,JWL)/ZRRR(JI,JWL)
!
   ENDDO
ENDDO
!
!---------------------------------------------------------------------------------
! 2- Compute surface diffuse albedo ZRDF
!---------------------------------------------------------------------------------
!
! * Diffuse albedo from Jin et al., 2006 (Eq 5b) 
!
DO JWL=1,NNWL
   DO JI=1,INI
      ZRDF(JI,JWL) = -0.1479 + 0.1502*XAKREFM(JWL) - 0.0176*ZSIG(JI)*XAKREFM(JWL)
   ENDDO
ENDDO
!
!---------------------------------------------------------------------------------
! 3- Compute direct water-leaving albedo ZRW
!---------------------------------------------------------------------------------
!
! * Chlorophyll derived values
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!when Chlorophyll will be coupled ZCHL(1) SHOULD BE ZCHL(:)
ZWORK4(:)= EXP(LOG(ZCHL(1))*0.65)
ZWORK5(:)= LOG10(ZCHL(1))  
ZBP550(:)= 0.416 * EXP(LOG(ZCHL(1))*0.766) 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! * Direct reflectance partitioning based on Morel & Gentilli 1991
!
ZR22(:)=0.48168549-0.014894708*ZSIG(:)-0.20703885*ZSIG(:)*ZSIG(:)
!
DO JWL=1,NNWL
   DO JI=1,INI
!
!     Determine absorption and backscattering
!     coefficients to determine reflectance below the surface (Ro) once for all
      ZAP(JI,JWL) = 0.06*XAKACHL(JWL)*ZWORK4(JI) + 0.2*(XAW440+0.06*ZWORK4(JI))*ZYLMD(JWL)
!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !when Chlorophyll will be coupled this condition should be activated
      !IF(ZCHL(JI)>2.)THEN
      !  ZBBP(JI,JWL)=(0.002+0.01*(0.5-0.25*ZWORK5(JI)))*ZBP550(JI)
      !ELSEIF(ZCHL(JI) < 0.02)THEN
      !  ZBBP(JI,JWL)=0.019*(550./XAKWL(JWL))*ZBP550(JI)
      !ELSE
        ZNU   (JI)     = 0.5*(ZWORK5(JI)-0.3)
        ZWORK1(JI)     = EXP(ZNU(JI)*LOG(XAKWL(JWL)/550.))
        ZBBP  (JI,JWL) = (0.002+0.01*(0.5-0.25*ZWORK5(JI))*ZWORK1(JI))*ZBP550(JI)
      !ENDIF
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!     Morel-Gentili(1991), Eq (12)
      ZHB(JI,JWL)=0.5*XAKBW(JWL)/(0.5*XAKBW(JWL)+ZBBP(JI,JWL))
!
!     Use Morel 91 formula to compute the direct reflectance below the surface
      ZWORK2(JI)=(0.5*XAKBW(JWL)+ZBBP(JI,JWL))/(XAKAW3(JWL)+ZAP(JI,JWL))
      ZWORK3(JI)=(0.6279-0.2227*ZHB(JI,JWL)-0.0513*ZHB(JI,JWL)*ZHB(JI,JWL) &
                -0.3119*ZCOSZEN(JI))+0.2465*ZHB(JI,JWL)*ZCOSZEN(JI)
      ZR00(JI,JWL)=ZWORK2(JI)*ZWORK3(JI)
!
!     water-leaving albedo
      ZRW(JI,JWL)=ZR00(JI,JWL)*(1.-ZR22(JI))/(1.-ZR00(JI,JWL)*ZR22(JI))
!
   ENDDO
ENDDO
!
!---------------------------------------------------------------------------------
! 4- Compute diffuse water-leaving albedo ZRWDF
!---------------------------------------------------------------------------------
!
ZWORK4(:)=0.0152-1.7873*ZUE(:)+6.8972*ZUE2(:)-8.5778*ZUE3(:)+4.071*ZSIG(:)-7.6446*ZUE(:)*ZSIG(:)
ZWORK5(:)=EXP(0.1643-7.8409*ZUE(:)-3.5639*ZUE2(:)-2.3588*ZSIG(:)+10.0538*ZUE(:)*ZSIG(:))
!
DO JWL=1,NNWL
   DO JI=1,INI
!
!     as previous water-leaving computation but assumes a uniform incidence of shortwave at surface (ue)
!
      ZWORK1(JI)=SQRT(1.0-(1.0-ZUE2(JI))/ZAKREFM2(JWL))    
!
      ZWORK2(JI)=(ZWORK1(JI)-XAKREFM(JWL)*ZUE(JI))/(ZWORK1(JI)+XAKREFM(JWL)*ZUE(JI))
      ZWORK2(JI)=ZWORK2(JI)*ZWORK2(JI)
      ZWORK3(JI)=(ZUE(JI)-XAKREFM(JWL)*ZWORK1(JI))/(ZUE(JI)+XAKREFM(JWL)*ZWORK1(JI))
      ZWORK3(JI)=ZWORK3(JI)*ZWORK3(JI)   
!
      ZRR0(JI,JWL)=0.50*(ZWORK2(JI)+ZWORK3(JI))
!
      ZWORK2(JI)=(ZWORK1(JI)-1.34*ZUE(JI))/(ZWORK1(JI)+1.34*ZUE(JI))
      ZWORK2(JI)=ZWORK2(JI)*ZWORK2(JI)
      ZWORK3(JI)=(ZUE(JI)-1.34*ZWORK1(JI))/(ZUE(JI)+1.34*ZWORK1(JI))
      ZWORK3(JI)=ZWORK3(JI)*ZWORK3(JI)   
!
      ZRRR(JI,JWL)=0.50*(ZWORK2(JI)+ZWORK3(JI))
      ZR11DF(JI)=ZRR0(JI,JWL)-ZWORK4(JI)*ZWORK5(JI)*ZRR0(JI,JWL)/ZRRR(JI,JWL)
!
!     Use Morel 91 formula to compute the diffuse
!     reflectance below the surface
      ZWORK2(JI)=(0.5*XAKBW(JWL)+ZBBP(JI,JWL))/(XAKAW3(JWL)+ZAP(JI,JWL))
      ZWORK3(JI)=0.6279-0.2227*ZHB(JI,JWL)-0.0513*ZHB(JI,JWL)*ZHB(JI,JWL) &
                -0.3119*ZUE(JI)+0.2465*ZHB(JI,JWL)*ZUE(JI)
      ZR00(JI,JWL)=ZWORK2(JI)*ZWORK3(JI)
!
!     diffuse albedo
      ZRWDF(JI,JWL)=ZR00(JI,JWL)*(1.-ZR22(JI))*(1.-ZR11DF(JI))/(1.-ZR00(JI,JWL)*ZR22(JI))
!
  ENDDO
ENDDO 
!
!---------------------------------------------------------------------------------
! 5- OSA estimation
!---------------------------------------------------------------------------------
!
! partitionning direct and diffuse albedo and accounts for foam spectral properties
!
DO JWL=1,NNWL
   DO JI=1,INI
      PDIR_ALB(JI) = PDIR_ALB(JI) + XFRWL(JWL) * ((1.-ZFWC(JI))*(ZR11(JI,JWL)+ZRW  (JI,JWL))+ZFWC(JI)*XRWC(JWL))
      PSCA_ALB(JI) = PSCA_ALB(JI) + XFRWL(JWL) * ((1.-ZFWC(JI))*(ZRDF(JI,JWL)+ZRWDF(JI,JWL))+ZFWC(JI)*XRWC(JWL))
  ENDDO 
ENDDO 
!
IF (LHOOK) CALL DR_HOOK('ALBEDO_OSAv1',1,ZHOOK_HANDLE)
!
!-------------------------------------------------------------------------------
!
END SUBROUTINE ALBEDO_OSAv1
