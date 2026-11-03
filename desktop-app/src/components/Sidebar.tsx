import React from 'react';
import {
  ShieldAlert,
  Activity,
  Flame,
  Cpu,
  Settings,
  Server,
  Radio,
  Wifi,
  Layers,
} from 'lucide-react';
import { ActiveTab, Telemetry } from '../types';

interface SidebarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  telemetry: Telemetry;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  telemetry,
}) => {
  const navItems: { id: ActiveTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    {
      id: 'monitor',
      label: 'Neural Defense 2D',
      icon: <Activity className="w-4 h-4" />,
    },
    {
      id: 'packets',
      label: 'Packets & Network',
      icon: <Radio className="w-4 h-4" />,
      badge: telemetry.total_packets > 0 ? `${telemetry.total_packets}` : 'LIVE',
    },
    {
      id: 'firewall',
      label: 'Live Firewall',
      icon: <ShieldAlert className="w-4 h-4" />,
      badge: telemetry.total_flows > 0 ? `${telemetry.total_flows}` : undefined,
    },
    {
      id: 'attack',
      label: 'Attack Hub (k6)',
      icon: <Flame className="w-4 h-4 text-amber-400" />,
      badge: 'DEMO',
    },
    {
      id: 'studio',
      label: 'Model Architecture',
      icon: <Cpu className="w-4 h-4" />,
    },
    {
      id: 'operations',
      label: 'System Operations',
      icon: <Server className="w-4 h-4" />,
    },
    {
      id: 'settings',
      label: 'Defense Settings',
      icon: <Settings className="w-4 h-4" />,
    },
  ];

  return (
    <aside className="w-72 bg-[#0B0F17] border-r border-slate-800 flex flex-col justify-between shrink-0 select-none">
      {/* Brand Header */}
      <div>
        <div className="p-5 border-b border-slate-800">
          <div className="font-bold tracking-wider text-base text-white flex items-center gap-1.5 font-mono">
            PREDATOR<span className="text-emerald-400">.NET</span>
          </div>
          <div className="text-[10px] text-slate-400 font-mono tracking-wider uppercase mt-0.5">
            AI Cyber Defense Suite
          </div>
        </div>


        {/* Live Hardware Telemetry Indicator */}
        <div className="mx-4 my-4 p-3 bg-[#111827] rounded-xl border border-slate-800 text-xs font-mono shadow-sm">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-slate-400 flex items-center gap-1.5 text-[11px]">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              ANN STATUS
            </span>
            <span className="text-emerald-400 font-bold text-[11px]">ACTIVE DEFENSE</span>
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>ACCELERATOR:</span>
            <span className="text-cyan-400 font-semibold truncate max-w-[130px]">RTX 4060 GPU</span>
          </div>
        </div>

        {/* Nav Links */}
        <nav className="px-3 space-y-1">
          {navItems.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-lg text-xs font-medium font-mono transition-all duration-150 ${
                  isActive
                    ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-center gap-3">
                  {item.icon}
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold ${
                      item.badge === 'DEMO'
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                        : item.badge === 'LIVE'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        : 'bg-slate-800 text-slate-300'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Target Server Indicator Footer */}
      <div className="p-4 border-t border-slate-800 bg-[#0E131F]">
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 mb-1">
          <Wifi className="w-3.5 h-3.5 text-cyan-400" />
          <span>TARGET ENDPOINT</span>
        </div>
        <div className="text-xs font-mono text-white bg-[#111827] px-2.5 py-1.5 rounded border border-slate-800 flex items-center justify-between">
          <span className="text-cyan-300 font-bold truncate">
            http://{telemetry.target_ip || '10.5.128.179'}:{telemetry.target_port || 8080}/
          </span>
        </div>
        <div className="mt-2 text-[10px] text-slate-400 flex items-center justify-between font-mono">
          <span>THREATS BLOCKED:</span>
          <span className="text-rose-400 font-bold">{telemetry.total_threats_blocked || 0}</span>
        </div>
      </div>
    </aside>
  );
};
