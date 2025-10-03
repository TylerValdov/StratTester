from celery import Task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.data_service import DataService
from app.services.ai_signals import AISignalsService
from app.services.backtester import BacktesterService
from app.services.lstm_trainer import LSTMTrainer
from app.models.backtest_result import BacktestResult
from app.models.strategy import Strategy
from app.models.lstm_model import LSTMModel
from app.models.user import User  # Import User model for relationships


# Create synchronous database session for Celery tasks
# (Celery workers can't use async)
sync_engine = create_engine(
    settings.DATABASE_URL.replace('+asyncpg', '').replace('postgresql', 'postgresql+psycopg2')
)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


class BacktestTask(Task):
    """Custom Celery task with progress tracking"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure"""
        print(f"Task {task_id} failed: {exc}")

        # Update database
        db = SyncSessionLocal()
        try:
            backtest = db.query(BacktestResult).filter(BacktestResult.task_id == task_id).first()
            if backtest:
                backtest.status = "FAILURE"
                backtest.error_message = str(exc)
                db.commit()

            # Also check for LSTM model training failure
            strategy_id = args[0] if args else None
            if strategy_id:
                lstm_model = db.query(LSTMModel).filter(LSTMModel.strategy_id == strategy_id).first()
                if lstm_model and lstm_model.status == "TRAINING":
                    lstm_model.status = "FAILURE"
                    lstm_model.error_message = str(exc)
                    db.commit()
        finally:
            db.close()


@celery_app.task(bind=True, base=BacktestTask, name="run_backtest_task")
def run_backtest_task(self, strategy_id: int):
    """
    Main Celery task for running backtests.

    Args:
        strategy_id: ID of the strategy to backtest

    Returns:
        Dictionary with backtest results
    """
    db = SyncSessionLocal()

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Initializing...'})

        # Fetch strategy from database
        strategy = db.query(Strategy).filter(Strategy.id == strategy_id).first()
        if not strategy:
            raise ValueError(f"Strategy with ID {strategy_id} not found")

        # Update backtest status
        backtest = db.query(BacktestResult).filter(
            BacktestResult.strategy_id == strategy_id,
            BacktestResult.task_id == self.request.id
        ).first()

        if backtest:
            backtest.status = "PROGRESS"
            db.commit()

        self.update_state(state='PROGRESS', meta={'current': 10, 'total': 100, 'status': 'Fetching market data...'})

        # Fetch historical data
        data_service = DataService()
        price_data = data_service.fetch_historical_data(
            ticker=strategy.ticker,
            start_date=strategy.start_date,
            end_date=strategy.end_date
        )

        if price_data.empty:
            raise ValueError(f"No price data available for {strategy.ticker}")

        # Generate AI signals if needed
        config = strategy.config
        use_prediction = config.get('use_prediction', False)
        lstm_config = config.get('lstm_config')

        ai_signals = None
        lstm_model_info = None

        if use_prediction and lstm_config:
            # Train LSTM model first
            self.update_state(state='PROGRESS', meta={'current': 30, 'total': 100, 'status': 'Training LSTM model...'})

            # Check if LSTM model already exists
            lstm_model = db.query(LSTMModel).filter(LSTMModel.strategy_id == strategy_id).first()

            if not lstm_model:
                # Create new LSTM model entry
                lstm_model = LSTMModel(
                    strategy_id=strategy_id,
                    ticker=strategy.ticker,
                    lstm_config=lstm_config,
                    training_start_date=strategy.start_date,
                    training_end_date=strategy.end_date,
                    status="TRAINING"
                )
                db.add(lstm_model)
                db.commit()
                db.refresh(lstm_model)

            # Update status
            lstm_model.status = "TRAINING"
            db.commit()

            # Train the model
            trainer = LSTMTrainer()

            def progress_callback(progress, message):
                # Map training progress to 30-70% of total
                total_progress = 30 + int(progress * 0.4)
                self.update_state(state='PROGRESS', meta={'current': total_progress, 'total': 100, 'status': message})

            try:
                training_results = trainer.train_model(
                    price_data=price_data,
                    lstm_config=lstm_config,
                    model_id=lstm_model.id,
                    progress_callback=progress_callback
                )

                # Update LSTM model with training results
                lstm_model.status = "SUCCESS"
                lstm_model.model_path = training_results['model_path']
                lstm_model.train_loss = training_results['train_loss']
                lstm_model.val_loss = training_results['val_loss']
                lstm_model.scaler_params = training_results['scaler_params']
                lstm_model.training_history = training_results['training_history']
                lstm_model.trained_at = datetime.utcnow()
                db.commit()

                # Prepare model info for prediction
                lstm_model_info = {
                    'model_path': training_results['model_path'],
                    'scaler_params': training_results['scaler_params'],
                    'lookback_period': lstm_config['lookback_period']
                }

            except Exception as e:
                lstm_model.status = "FAILURE"
                lstm_model.error_message = str(e)
                db.commit()
                raise e

            self.update_state(state='PROGRESS', meta={'current': 70, 'total': 100, 'status': 'Generating predictions from trained model...'})

            # Generate AI signals using trained model
            ai_service = AISignalsService()
            ai_signals = ai_service.get_combined_signals(
                price_data=price_data,
                use_prediction=use_prediction,
                lstm_model=lstm_model_info
            )

        elif use_prediction:
            # Use simplified technical approach if no LSTM config
            self.update_state(state='PROGRESS', meta={'current': 30, 'total': 100, 'status': 'Generating AI signals...'})
            ai_service = AISignalsService()
            ai_signals = ai_service.get_combined_signals(
                price_data=price_data,
                use_prediction=use_prediction
            )

        self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100, 'status': 'Running backtest simulation...'})

        # Run backtest
        initial_capital = config.get('initial_capital', 100000.0)
        backtester = BacktesterService(initial_capital=initial_capital)

        results = backtester.run_backtest(
            price_data=price_data,
            config=config,
            ai_signals=ai_signals
        )

        self.update_state(state='PROGRESS', meta={'current': 85, 'total': 100, 'status': 'Calculating benchmarks...'})

        # Calculate buy-and-hold benchmark for the ticker
        buy_hold_benchmark = backtester.calculate_benchmark(
            price_data=price_data,
            initial_capital=initial_capital
        )

        # Fetch SPY data for benchmark comparison
        spy_benchmark = None
        spy_return = None
        try:
            spy_data = data_service.fetch_historical_data(
                ticker='SPY',
                start_date=strategy.start_date,
                end_date=strategy.end_date
            )
            if not spy_data.empty:
                spy_result = backtester.calculate_benchmark(
                    price_data=spy_data,
                    initial_capital=initial_capital
                )
                spy_benchmark = spy_result['equity_curve']
                spy_return = spy_result['total_return']
        except Exception as e:
            print(f"Warning: Could not fetch SPY benchmark data: {e}")

        self.update_state(state='PROGRESS', meta={'current': 90, 'total': 100, 'status': 'Saving results...'})

        # Update backtest result in database
        if backtest:
            backtest.status = "SUCCESS"
            backtest.final_balance = results['final_balance']
            backtest.total_return = results['total_return']
            backtest.sharpe_ratio = results['sharpe_ratio']
            backtest.max_drawdown = results['max_drawdown']
            backtest.total_trades = results['total_trades']
            backtest.winning_trades = results['winning_trades']
            backtest.losing_trades = results['losing_trades']
            backtest.trade_log = results['trade_log']
            backtest.equity_curve = results['equity_curve']

            # Store benchmark data
            backtest.buy_hold_benchmark = buy_hold_benchmark['equity_curve']
            backtest.buy_hold_return = buy_hold_benchmark['total_return']
            backtest.spy_benchmark = spy_benchmark
            backtest.spy_return = spy_return

            backtest.completed_at = datetime.utcnow()

            db.commit()

        self.update_state(state='SUCCESS', meta={'current': 100, 'total': 100, 'status': 'Complete!'})

        return {
            'status': 'SUCCESS',
            'backtest_id': backtest.id if backtest else None,
            'results': results
        }

    except Exception as e:
        # Update backtest with error
        if backtest:
            backtest.status = "FAILURE"
            backtest.error_message = str(e)
            db.commit()

        raise e

    finally:
        db.close()
