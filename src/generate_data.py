import numpy as np
from matplotlib import pyplot as plt
from astropy.cosmology import Planck18
from astropy import units
from lensing_hk_24 import kappa_from
from cluster_toolkit import deltasigma

'''

The function to generate data. 

example_generate_samples = generate nsamples (with noise) log(radius) (X) and log(density) (Y) 
                                 for a given halo mass and redshift
                        
example_plot_profiles = generate many profiles (without noise) for different halo mass
                        and redshifts
                        
NFW = the workhorse that does the data generation

mass_concentration = function required by NFW

'''

def NFW_projected(
        radius = np.logspace( 0, 4, 10000)*units.kpc,
        nsamples=1000, 
        halo_mass=10**15*units.Msun,
        cluster_redshift=0.25,
        add_scatter=False
    ):
    

    concentration = mass_concentration( 
        halo_mass, h=0.7, add_scatter=add_scatter 
    )
    
    omega_m = Planck18.Om(cluster_redshift)
    
    Sigma_nfw = deltasigma.Sigma_nfw_at_R(
        radius.to(units.Mpc),
        halo_mass.value, 
        concentration, 
        omega_m
    )
    
    return radius, Sigma_nfw

def example_generate_samples( 
    nsamples=1000, 
    halo_mass=10**15*units.Msun,
    cluster_redshift=0.25,
    projected=False,
    ):
    
    
    profiles = []
    radius = []
    for i in range( nsamples ):
        
        if not projected:
            this_radius, density = NFW(
                halo_mass,
                cluster_redshift=cluster_redshift,
                add_scatter=True,
                return_values=True
            )
        else:
            NFW_projected(
                radius = np.logspace( 0, 4, 10000)*units.kpc,
                nsamples=1000, 
                halo_mass=10**15*units.Msun,
                cluster_redshift=0.25,
                add_scatter=False
                )
        radius.append( np.log10(this_radius) )
        profiles.append( np.log10(density) )
    
    profiles = np.vstack( profiles )
    radius = np.vstack( radius )
    
    return radius, profiles
    
def example_plot_profiles( ):
    
    #Make a plot
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    
    # Loop over many halo mass ranges
    mass_range = np.logspace(14, 15.5, 10)*units.Msun
    # Loop over many redshift ( how old the halo is)
    redshift_range = np.linspace(0,1,10)
    
    for halo_mass in mass_range:
        for halo_redshift in redshift_range:
            
            radius, density = NFW( 
                halo_mass, 
                cluster_redshift=halo_redshift,
                return_values=True
            )
                
            ax.plot( 
                np.log10( radius ), 
                np.log10( density),
                '--'
            )
    plt.xlabel(r"$\log($Radius$ / kpc )$")
    plt.ylabel(r"$\log(\rho/ (M_\odot/$kpc$^3$))")
    plt.savefig("data_generated.png")

    
    
def NFW( 
    halo_mass, 
    cluster_redshift=0.25,
    radius = np.logspace( 0, 4, 10000)*units.kpc,
    overdensity=200.,
    add_scatter=False,
    return_values=False
):
    '''
    Purpose - generate the density profile of an NFW
    
    Inputs:
        halo_mass : the mass of the main in units of units.Msun
    
    Keywords:
        cluster_redshift : float, a value between 0 and 1 of how far away the cluster is.
                    The profile does depend on this so would be good to include as parameter.
        radius : array, the radius range which we calculate the profile over (SHOULD NOT CHANGE)
        overdensity : float, the factor over which we calcualte the densty of the universe (DO NOT CHANGE)
        add_scatter : bool, add scatter / noise to the density profiles
        return_values : since everything is in Units to be consistent, this bool returns only values to make
                    them more compatible with other algorithms that dont expect this.
        
    All numbers and coefficeints are from https://arxiv.org/pdf/0805.1926
    
    
    '''
    
    # Get the concentration of the halo
    concentration = mass_concentration( 
        halo_mass, h=0.7, add_scatter=add_scatter )
    
    # Density of the universe at the redshift of the cluster
    critical_density = Planck18.critical_density(cluster_redshift).to(units.Msun/units.kpc**3)
    
    # What is the radius of the cluster
    virial_radius = (
        halo_mass / (4./3*np.pi*critical_density*overdensity)
        
    )**(1./3.)
    
    # The scale radius of the halo which is a parameter of the NFW
    scale_radius = virial_radius / concentration
    
    # The normalisation of the halos
    delta_denom = (
         np.log( 1 + concentration ) - concentration / ( 1. + concentration) 
    )
    
    delta_c = overdensity  / 3. * concentration**3 /  delta_denom
    
    normalisation = critical_density * delta_c
    
    # Now get the radius scaled to the scale radius
    X = radius / scale_radius
    
    # Now work out the density
    rho = normalisation / (
        X*(1+X)**2
    )

    #return in units of kpc and Msun / kpc**3
    if return_values:
        return radius.value, rho.value
    else:
        return radius, rho


def mass_concentration( halo_mass, h=0.7, add_scatter=False ):
    '''
    
    What is the concentration of the halo for a given halo mass?
    According to https://arxiv.org/pdf/0805.1926
    '''
    
    A = 0.830 
    B = 0.098 
    
    log_c200 = A - B * np.log10(halo_mass/(1e12*units.Msun/h))
    
    # Add some dispersion to this if required.
    if add_scatter:
        dispersion = 0.015
    else:
        dispersion = 0.
        
    scatter = dispersion*np.random.normal()
    
    observed_conc = log_c200 + scatter
    
    return 10**observed_conc


if __name__ == '__main__':
    example_plot_profiles()