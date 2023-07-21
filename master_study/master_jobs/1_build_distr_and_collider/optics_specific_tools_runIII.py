from xmask.lhc import install_errors_placeholders_hllhc


def build_sequence(
    mad,
    mylhcbeam,
    ignore_cycling=False,
    **kwargs,
):
    # Select beam
    mad.input(f"mylhcbeam = {mylhcbeam}")

    mad.input(f"""
      ! Get the toolkit
      call,file=
        "acc-models-lhc/toolkit/macro.madx";
      """)

    mad.input(f"""
      ! Build sequence
      option, -echo,-warn,-info;
      if (mylhcbeam==4){{
        call,file="acc-models-lhc/lhc_acc-models-lhc_b4.seq";
      }} else {{
        call,file="acc-models-lhc/lhc_acc-models-lhc.seq";
      }};
      option, -echo, warn,-info;
      """)

    mad.input("""
      ! Slice nominal sequence
      slicefactor=4;
      myslice_plus: macro = {
      if (MBX.4L2->l>0) {
        select, flag=makethin, clear;
        select, flag=makethin, class=mb, slice=2;
        select, flag=makethin, class=mq, slice=2*slicefactor;
        select, flag=makethin, class=mqxa,  slice=16*slicefactor;  !old triplet
        select, flag=makethin, class=mqxb,  slice=16*slicefactor;  !old triplet
        select, flag=makethin, class=mqxc,  slice=16*slicefactor;  !new mqxa (q1,q3)
        select, flag=makethin, class=mqxd,  slice=16*slicefactor;  !new mqxb (q2a,q2b)
        select, flag=makethin, class=mqxfa, slice=16*slicefactor;  !new (q1,q3 v1.1)
        select, flag=makethin, class=mqxfb, slice=16*slicefactor;  !new (q2a,q2b v1.1)
        select, flag=makethin, class=mbxa,  slice=4;   !new d1
        select, flag=makethin, class=mbxf,  slice=4;   !new d1 (v1.1)
        select, flag=makethin, class=mbrd,  slice=4;   !new d2 (if needed)
        select, flag=makethin, class=mqyy,  slice=4*slicefactor;   !new q4
        select, flag=makethin, class=mqyl,  slice=4*slicefactor;   !new q5
        select, flag=makethin, class=mbh,   slice=4;   !11T dipoles
        select, flag=makethin, pattern=mbx\.,    slice=4;
        select, flag=makethin, pattern=mbrb\.,   slice=4;
        select, flag=makethin, pattern=mbrc\.,   slice=4;
        select, flag=makethin, pattern=mbrs\.,   slice=4;
        select, flag=makethin, pattern=mbh\.,    slice=4;
        select, flag=makethin, pattern=mqwa\.,   slice=4*slicefactor;
        select, flag=makethin, pattern=mqwb\.,   slice=4*slicefactor;
        select, flag=makethin, pattern=mqy\.,    slice=4*slicefactor;
        select, flag=makethin, pattern=mqm\.,    slice=4*slicefactor;
        select, flag=makethin, pattern=mqmc\.,   slice=4*slicefactor;
        select, flag=makethin, pattern=mqml\.,   slice=4*slicefactor;
        select, flag=makethin, pattern=mqtlh\.,  slice=2*slicefactor;
        select, flag=makethin, pattern=mqtli\.,  slice=2*slicefactor;
        select, flag=makethin, pattern=mqt\.  ,  slice=2;
        !thin lens
        if (version >= 50208) { !
          option rbarc=false; beam;
          use,sequence=lhcb1; makethin,sequence=lhcb1,makedipedge=true,style=teapot;
          use,sequence=lhcb2; makethin,sequence=lhcb2,makedipedge=true,style=teapot;
          option rbarc=true;
        } else {
          beam;
          use,sequence=lhcb1; makethin, sequence=lhcb1,style=teapot;
          use,sequence=lhcb2; makethin, sequence=lhcb2,style=teapot;
        }
      } else {
        print, text="Sequence is already thin";
      };
        is_thin=1;
      };
              
      exec, myslice_plus;
      """)

    mad.input(f"""
    nrj=450;
    beam,particle=proton,sequence=lhcb1,energy=nrj,npart=1.6E11,sige=4.5e-4;
    beam,particle=proton,sequence=lhcb2,energy=nrj,bv = -1,npart=1.6E11,sige=4.5e-4;
    """)

    if not ignore_cycling:
        mad.input("""
        !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
        if (mylhcbeam<3){
        seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
        };
        seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;
        """)

    mad.input("""
        ! Set twiss formats for MAD-X parts (macro from opt. toolkit)
        exec, twiss_opt;
        """)


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    # A knob redefinition
    mad.input("on_alice := s;")
    mad.input("on_lhcb := on_lhcb_normalized * 7000./nrj;")
