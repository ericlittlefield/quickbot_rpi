#!/usr/bin/python
"""
@brief Run QuickBot class for Beaglebone Black

@author Josip Delic (delijati.net)
@author Rowland O'Flaherty (rowlandoflaherty.com)
@date 02/07/2014
@version: 1.0
@copyright: Copyright (C) 2014, see the LICENSE file
"""

import sys
import argparse

def get_ip(iface = 'eth0'):
    ifreq = struct.pack('16sH14s', iface, socket.AF_INET, '\x00'*14)
    try:
        res = fcntl.ioctl(sockfd, SIOCGIFADDR, ifreq)
    except:
        return None
    ip = struct.unpack('16sH2x4s8x', res)[2]
    return socket.inet_ntoa(ip)



DESCRIPTION = ""
RTYPES = ('quick_v2', 'quick_v1', 'ultra', 'quick_rpi')


def main(options):
    """ Main function """
    print "Running XBot"

    print 'Running XBot Program'
    print '    Base IP: ', options.ip
    print '    Robot IP: ', options.rip
    print '    Robot Type: ', options.rtype

    if options.rtype == 'quick_v2':
        import xbots.quickbot_v2
        qb = xbots.quickbot_v2.QuickBot(options.ip, options.rip)
    elif options.rtype == 'quick_v1':
        import xbots.quickbot_v1
        qb = xbots.quickbot_v1.QuickBot(options.ip, options.rip)
    elif options.rtype == 'ultra':
        import xbots.ultrabot
        qb = xbots.ultrabot.UltraBot(options.ip, options.rip)
    elif option.rtype == 'quick_rpi':
        qb = xbots.quickbot_rpi.QuickBot_rpi(options.ip, option.rip)
    else:
        raise Exception("No valid robot target provided")

    qb.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        '--ip', '-i',
        default='192.168.7.1',
        help="Computer ip (base ip)")
    parser.add_argument(
        '--rip', '-r',
        default=get_ip(),
        help="robot ip")
    parser.add_argument(
        '--rtype', '-t',
        default='quick_rpi',
        help="Type of robot (%s)" % '|'.join(RTYPES))

    options = parser.parse_args()
    if options.rtype not in RTYPES:
        print "Chosen type not exists use (%s)" % '|'.join(RTYPES)
        sys.exit(0)
    main(options)
