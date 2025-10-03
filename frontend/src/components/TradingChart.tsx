import { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';
import type { Trade } from '../types';

interface PriceData {
  time: string;
  value: number;
}

interface TradingChartProps {
  priceData: PriceData[];
  trades: Trade[];
  height?: number;
}

export default function TradingChart({ priceData, trades, height = 400 }: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#333',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
      },
    });

    chartRef.current = chart;

    // Add price line series
    const lineSeries = chart.addLineSeries({
      color: '#2962FF',
      lineWidth: 2,
    });

    seriesRef.current = lineSeries;

    // Set price data
    if (priceData.length > 0) {
      lineSeries.setData(priceData);
    }

    // Add trade markers
    if (trades.length > 0) {
      const markers = trades.map((trade) => ({
        time: trade.date,
        position: trade.action === 'BUY' ? 'belowBar' as const : 'aboveBar' as const,
        color: trade.action === 'BUY' ? '#26a69a' : '#ef5350',
        shape: trade.action === 'BUY' ? 'arrowUp' as const : 'arrowDown' as const,
        text: `${trade.action} @ $${trade.price.toFixed(2)}`,
      }));

      lineSeries.setMarkers(markers);
    }

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [priceData, trades, height]);

  return <div ref={chartContainerRef} className="w-full" />;
}
