interface PromoModalProps {
  open: boolean;
  onClose: () => void;
}

export function PromoModal({ open, onClose }: PromoModalProps): JSX.Element | null {
  if (!open) {
    return null;
  }

  return (
    <div className="promo-backdrop" role="presentation" onClick={onClose}>
      <section
        className="promo-modal"
        aria-labelledby="promo-title"
        aria-describedby="promo-description"
        role="dialog"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
      >
        <p className="promo-tag">High-intent user detected</p>
        <h2 id="promo-title">Complete the look with our Comfort+ Nursing Pack</h2>
        <p id="promo-description">
          Add one nursing camisole and one support bra now to unlock 15% bundle savings
          before checkout.
        </p>
        <div className="promo-actions">
          <button type="button" className="promo-primary" onClick={onClose}>
            Add Bundle
          </button>
          <button type="button" className="promo-secondary" onClick={onClose}>
            Continue Browsing
          </button>
        </div>
      </section>
    </div>
  );
}
